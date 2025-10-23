
from pathlib import Path
import json
import subprocess
import torch
import h5py
import traceback
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from model.model_abmil import ABMIL_Surv_PG
import pandas as pd
import numpy as np

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
LOG_FILE = OUTPUT_PATH / "debug.log"

class ClinicalFeatureExtractor:
    def __init__(self, numeric_cols=None, categorical_cols=None):
        self.numeric_cols = numeric_cols or ["age", "no_instillations"]
        self.categorical_cols = ["sex", "smoking", "tumor", "stage", "substage", "grade", "reTUR", "LVI", "variant", "EORTC", "BRS"]
        self.preprocessor = None

    def fit(self, clinical_dir, train_slide_ids):
        """
        Fit the preprocessor using only training JSON files.
        
        Args:
            clinical_dir (str): Path to the directory containing clinical JSON files.
            train_slide_ids (list[str]): List of training slide IDs, e.g., ["2A_001_HE", "2A_003_HE"].
        """
        data = []
        for slide_id in train_slide_ids:
            slide_base = slide_id.replace("_HE", "_CD.json")  # convert to your clinical file naming
            json_path = os.path.join(clinical_dir, slide_base)
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data.append(json.load(f))
            else:
                print(f"Warning: Clinical file not found: {json_path}")

        df = pd.DataFrame(data)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols)
            ]
        )
        self.preprocessor.fit(df)
        return self

    def transform(self, clinical_json_path):
        """Transform a single clinical JSON file into a normalized feature tensor."""
        clinic_raw = json.load(open(clinical_json_path, "r"))
        clinic_raw.pop("Time_to_prog_or_FUend", None)
        clinic_raw.pop("progression", None)     
        df = pd.DataFrame([clinic_raw])
        features = self.preprocessor.transform(df)
        features = features.toarray() if hasattr(features, "toarray") else features
        return features

    def save(self, path):
        """Save the fitted preprocessor to disk."""
        joblib.dump((self.numeric_cols, self.categorical_cols, self.preprocessor), path)

    def load(self, path):
        """Load the fitted preprocessor from disk."""
        self.numeric_cols, self.categorical_cols, self.preprocessor = joblib.load(path)
        return self



def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(str(msg) + "\n")

def list_files(path: Path):
    log(f"Listing files in {path}:")
    for p in path.rglob("*"):
        log(f"  {p}")

def run():
    try:
        # --- List files in /input and /output before starting ---
        list_files(INPUT_PATH)
        list_files(OUTPUT_PATH)

        # Paths for input
        wsi_path = INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi"
        mask_path = INPUT_PATH / "images/tissue-mask"
        cli_path = INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-recurrence-patients.json"
        rna_path = INPUT_PATH / "bulk-rna-seq-bladder-cancer.json"

        _show_torch_cuda_info()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        patches_dir = OUTPUT_PATH / "Bladder_10x"
        features_dir = patches_dir / "feat_uni"
        h5_dir = features_dir / "h5_files"

        # --- 1. Extract patches ---
        log("Starting patch extraction...")
        with open(LOG_FILE, "a") as lf:
            subprocess.run([
                "python", "ext_feat/create_patches_fp.py",
                "--source", str(wsi_path),
                "--source_mask", str(mask_path),
                "--save_dir", str(patches_dir),
                "--patch_level", "1",
                "--patch_size", "224",
                "--step_size", "224",
                "--seg", "--patch", "--stitch", "--use_ostu"
            ], stdout=lf, stderr=lf, text=True)

        # --- 2. Extract features ---
        log("Starting feature extraction...")
        with open(LOG_FILE, "a") as lf:
            subprocess.run([
                "python", "ext_feat/extract_features.py",
                "--data_h5_dir", str(patches_dir),
                "--data_slide_dir", str(wsi_path),
                "--csv_path", str(patches_dir / "process_list_autogen.csv"),
                "--feat_dir", str(features_dir),
                "--batch_size", "64"
            ], stdout=lf, stderr=lf, text=True)

        # --- 3. Load model ---
        log("Loading model...")
        ckpt_path = Path("model/s_0_checkpoint.pt")
        model = ABMIL_Surv_PG(feat_type="uni").to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_clean = {k.replace('.module', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt_clean, strict=True)

        model.relocate()
        model.eval()

        # --- Clinical data ---
        clinic = torch.tensor(extractor.transform(cli_path), dtype=torch.float32, device=device)

        # --- RNA data ---
        with open("model/gene_order.json", "r") as f:
            gene_order = json.load(f)

        with open(rna_path, "r") as f:
            rna_dict = json.load(f)
            
        rna_vec = [rna_dict.get(k, 0.0) for k in gene_order]
        rna = torch.tensor(rna_vec, dtype=torch.float32, device=device).unsqueeze(0)

        # --- 4. Process H5 files ---
        h5_files = list(h5_dir.glob("*.h5"))
        if not h5_files:
            log(f"No .h5 files found in {h5_dir}")
            raise RuntimeError("No .h5 feature files produced.")

        h5_file_path = h5_files[0]  # current slide only
        log(f"Processing {h5_file_path}")
        with h5py.File(h5_file_path, 'r') as h5_file:
            features = h5_file['features'][:]

        data = torch.tensor(features, dtype=torch.float32, device=device)

        with torch.no_grad():
            risk, _ = model(data, clinic, rna)

        # Load KM curve
        km_data = joblib.load("model/km_curve.pkl")
        km_times = km_data["km_times"]
        km_probs = km_data["km_probs"]
        train_risks = km_data["train_risks"]

        # Compute predicted recurrence time
        combined_risks = np.append(train_risks, risk.item())
        rank = (combined_risks >= risk.item()).sum()  # higher risk → lower survival
        prob = 1.0 - (rank - 1) / len(combined_risks)
        idx = (np.abs(km_probs - prob)).argmin()
        pred_time = float(km_times[idx])
        pred_time = max(0.0, pred_time)

        with open(OUTPUT_PATH / "likelihood-of-bladder-cancer-recurrence.json", "w") as f:
            json.dump(pred_time, f, indent=4)

        log("Inference completed successfully.")
        return 0

    except Exception:
        with open(LOG_FILE, "a") as f:
            f.write("Fatal error:\n")
            f.write(traceback.format_exc())
        raise

def _show_torch_cuda_info():
    import torch
    log("=+=" * 10)
    log(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"Devices: {torch.cuda.device_count()}")
        log(f"Current: {torch.cuda.current_device()}")
        log(f"Props: {torch.cuda.get_device_properties(torch.cuda.current_device())}")
    log("=+=" * 10)

if __name__ == "__main__":
    extractor = ClinicalFeatureExtractor().load("model/clinical_preprocessor.pkl")
    raise SystemExit(run())
