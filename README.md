# CHIMERA Challenge Task 3

**Repository for Task 3 of the CHIMERA Challenge: Bladder Cancer Recurrence Prediction**  

This repository contains code, models, and instructions for reproducing experiments for Task 3 of the CHIMERA Challenge.

---

## 📌 Table of Contents
1. [Overview](#overview)  
2. [Data](#data)  
3. [Usage](#usage)   
4. [Training & Evaluation](#training--evaluation)  

---

## Overview
Task 3 focuses on **predicting patient survival risk** using multi-modal data, including **histopathology slides, RNA expression, and clinical features**.  

The pipeline includes:
- Patch-level feature extraction with the UNI model  
- Slide-level aggregation using ABMIL  
- RNA embedding compression via an RNA encoder  
- Fusion of slide, RNA, and clinical data with different fusion modules  
- Risk score prediction via a linear classifier  

---

## Data
Please download the CHIMERA Task 3 dataset from the official challenge website.

---

## Usage

# Patch Extraction
python create_patches_fp.py --source .../bladder-cancer-tissue-biopsy-wsi  --source_mask .../tissue-mask --save_dir ./Bladder_10x_ --patch_level 1 --patch_size 224 --step_size 224 --seg --patch

# Feature Extraction
python extract_features.py --data_h5_dir Bladder_10x --data_slide_dir .../bladder-cancer-tissue-biopsy-wsi --csv_path Bladder_10x/process_list_autogen.csv --feat_dir ./CHIM_Rec_ostu_10x/feat_uni --batch_size 256 --slide_ext .tif

# Training
python train.py --model_type pg --exp_code PG_10x_CLIN_RNA_3e-1 --reg 3e-1

# Evaluation
python eval.py --models_exp_code PG_10x_CLIN_RNA_3e-1_s2021 --save_exp_code PG_10x_CLIN_RNA_3e-1 --model_typ pg 

