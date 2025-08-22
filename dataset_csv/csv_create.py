
import os
import json
import csv

# Path to your data directory
data_dir = "/home/20215294/Data/CHIM_Rec/CHIM_Rec_ostu_10x/feat_uni/clinics"
output_csv = "bladder-rec.csv"

rows = []
for case_id in sorted(os.listdir(data_dir)):
    case_path = os.path.join(data_dir, case_id)

    json_file = case_path
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            info = json.load(f)
            time = info.get("Time_to_prog_or_FUend", None)
            prog = info.get("progression", None)

            slide_id = case_id.replace('_CD.json', '') + '_HE'
            rows.append([case_id.replace('_CD.json', ''), slide_id, time, prog])
    else:
        print(f"Warning: No JSON file found for {case_id.replace('_CD.json', '')}")

# Write the output CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case_id", "slide_id", "Time", "Progression"])
    writer.writerows(rows)

print(f"Saved {len(rows)} entries to {output_csv}")
