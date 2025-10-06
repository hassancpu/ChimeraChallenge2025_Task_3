# 🚀 CHIMERA Challenge Task 3

**Bladder Cancer Recurrence Prediction using Multi-modal Data**  

This repository contains code, models, and instructions for reproducing experiments for **Task 3 of the CHIMERA Challenge**.

---

## 📌 Table of Contents
1. [Overview](#overview)  
2. [Data](#data)  
3. [Full Pipeline Usage](#full-pipeline-usage)  
4. [Training & Evaluation](#training--evaluation)  
5. [License](#license)  

---

## 🧠 Overview
Task 3 focuses on **predicting patient survival risk** using multi-modal data:  
- **Histopathology slides**  
- **RNA expression**  
- **Clinical features**  

**Pipeline Highlights:**
- Patch-level feature extraction with **UNI model**  
- Slide-level aggregation using **ABMIL**  
- RNA embedding compression via **RNA encoder**  
- Fusion of slide, RNA, and clinical data with multiple **fusion modules**  
- Risk score prediction using a **linear classifier**  

---

## 📂 Data
The CHIMERA Task 3 dataset must be downloaded from the official challenge website.  
Organize the data as follows:  

```text
CHIM_Rec_ostu_10x/
├── pt_files/             # Patch feature files for each slide
├── seqs/                 # RNA sequences or embeddings
├── clinics/              # Clinical data files
gene_order.json           # Gene order reference for RNA data
clinical_preprocessor.pkl # Preprocessing object for clinical features
```

---

## 🛠 Full Pipeline Usage

Run the complete workflow for Task 3 in sequence:

```bash
# 1️⃣ Patch Extraction
python create_patches_fp.py \
    --source .../bladder-cancer-tissue-biopsy-wsi \
    --source_mask .../tissue-mask \
    --save_dir ./Bladder_10x_ \
    --patch_level 1 \
    --patch_size 224 \
    --step_size 224 \
    --seg \
    --patch

# 2️⃣ Feature Extraction
python extract_features.py \
    --data_h5_dir Bladder_10x \
    --data_slide_dir .../bladder-cancer-tissue-biopsy-wsi \
    --csv_path Bladder_10x/process_list_autogen.csv \
    --feat_dir ./CHIM_Rec_ostu_10x/feat_uni \
    --batch_size 256 \
    --slide_ext .tif

# Coords Extraction
python coord.py

# 3️⃣ Training
python train.py \
    --model_type pg \
    --exp_code PG_10x_CLIN_RNA_3e-1 \
    --reg 3e-1

# 4️⃣ Evaluation
python eval.py \
    --models_exp_code PG_10x_CLIN_RNA_3e-1_s2021 \
    --save_exp_code PG_10x_CLIN_RNA_3e-1 \
    --model_typ pg
```

---

## ⚖ License

This repository is licensed under MIT License.
