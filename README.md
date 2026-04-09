# 2025_Dacon_Winning_Solution

Official repository for our solution to **Jump AI(.py) 2025: 3rd AI Drug Discovery Competition**.

This repository is associated with our study:

**SMILES-Based IC50 Prediction for ASK1 Inhibitors through Machine Learning**

The goal of this project is to predict the bioactivity of **ASK1 inhibitors** from **SMILES strings** using a compact and practical machine learning pipeline. The repository includes data preprocessing, experiment notebooks, final model training, and inference code for molecular activity prediction.

---

## Overview

Bioactivity prediction is an important task in AI-driven drug discovery because it helps prioritize candidate compounds before costly experimental validation. In this project, we focused on predicting the inhibitory activity of molecules against **ASK1 (Apoptosis Signal-regulating Kinase 1)**.

This repository contains code for:

- data preprocessing and curation
- experiment-based model comparison
- final SVR-based model training
- inference for unseen molecules
- a competition-oriented activity prediction workflow

---

## Workflow

The overall pipeline of this repository is summarized below.

<p align="center">
  <img src="Figures/workflow.pdf" width="950"/>
</p>

<p align="center">
  <em>Figure 1. Overview of the proposed workflow for ASK1 inhibitor activity prediction. The pipeline consists of data collection from ChEMBL, PubChem, and CAS; preprocessing steps such as counter-ion removal, duplicate removal, and SMILES canonicalization; molecular representation using pre-trained encoders; model selection through comparison of 36 regression models with augmentation and tuning; and final IC50 prediction.</em>
</p>

A typical workflow consists of:

1. collecting and curating ASK1-related datasets  
2. preprocessing SMILES strings and activity labels  
3. conducting regression experiments and model comparisons  
4. training the final tuned prediction model  
5. performing inference on unseen molecules  

---

## Repository Structure

```bash
2025_Dacon/
├── Data/
│   ├── ChEMBL_ASK1(IC50).csv
│   └── Pubchem_ASK1.csv
├── Model/
│   ├── Model_Inference_SVR_Tuning.py
│   └── Model_Train_SVR_Tuning.py
├── Preprocessing/
│   └── Data_Preprocessing.ipynb
├── Experiments/
│   ├── experiments_preparation.ipynb
│   ├── experiments_01_all_models.ipynb
│   ├── experiments_02_svr_aug_vs_no_aug.ipynb
│   └── experiments_03_svr_tuned_vs_svr_aug_only.ipynb
├── figures/
│   └── workflow.png
├── Dacon.yaml
└── README.md
```

---

## Folder Description

### `Data/`
Contains example datasets used in this project.

- `ChEMBL_ASK1(IC50).csv`  
  ASK1-related activity data collected from ChEMBL

- `Pubchem_ASK1.csv`  
  ASK1-related activity data collected from PubChem

### `Preprocessing/`
Contains the notebook for dataset curation and preprocessing.

- `Data_Preprocessing.ipynb`  
  preprocessing and curation of raw molecular datasets

### `Experiments/`
Contains notebooks for experiment preparation and regression model comparison.

- `experiments_preparation.ipynb`  
  prepares data and settings for downstream experiments

- `experiments_01_all_models.ipynb`  
  compares multiple regression models

- `experiments_02_svr_aug_vs_no_aug.ipynb`  
  compares SVR with and without augmentation

- `experiments_03_svr_tuned_vs_svr_aug_only.ipynb`  
  compares tuned SVR against augmentation-only SVR

### `Model/`
Contains the final model scripts.

- `Model_Train_SVR_Tuning.py`  
  training script for the tuned SVR-based model

- `Model_Inference_SVR_Tuning.py`  
  inference script for prediction on unseen samples

### `Figures/`
Contains figures used in the README.

- `workflow.pdf`  
  overview figure of the full ASK1 prediction pipeline

### `Dacon.yaml`
Conda environment file for reproducing the software environment.

---

## Data Preparation

The preprocessing stage is used to prepare molecular datasets before model development.

Typical preprocessing steps include:

- loading molecular datasets from public sources
- removing counter-ions
- removing duplicated entries
- canonicalizing SMILES strings
- preparing activity labels for regression
- exporting processed datasets for downstream experiments

Run the preprocessing workflow through:

```bash
Preprocessing/Data_Preprocessing.ipynb
```

---

## Experiments

This repository includes experiment notebooks for systematic regression analysis.

The experiment stage includes:

- benchmarking multiple regression models
- comparing augmentation and non-augmentation settings
- comparing tuned SVR against simpler SVR-based baselines
- selecting the best encoder-model pair for final prediction

Run the experiment preparation notebook first if needed:

```bash
Experiments/experiments_preparation.ipynb
```

Then run the experiment notebooks as needed:

```bash
Experiments/experiments_01_all_models.ipynb
Experiments/experiments_02_svr_aug_vs_no_aug.ipynb
Experiments/experiments_03_svr_tuned_vs_svr_aug_only.ipynb
```

---

## Final Model Training

To train the final tuned model, run:

```bash
python Model/Model_Train_SVR_Tuning.py
```

This script is intended for the final regression model training used in the competition workflow.

---

## Inference

To perform inference with the trained model, run:

```bash
python Model/Model_Inference_SVR_Tuning.py
```

This script generates predictions for new molecular inputs.

---

## Environment Setup

The environment configuration is provided in `Dacon.yaml`.

Create the environment with:

```bash
conda env create -f Dacon.yaml
conda activate DACON_Verification
```

Depending on your local system, CUDA- or PyTorch-related packages may need small adjustments.

---

## Main Dependencies

This repository relies on several core packages, including:

- Python 3.11
- PyTorch
- Torch Geometric
- RDKit
- Transformers
- scikit-learn
- pandas
- NumPy
- tqdm
- OpenPyXL

---

## Reproducibility

For reproducibility, please use:

- the provided `Dacon.yaml`
- the preprocessing notebook in `Preprocessing/`
- the experiment notebooks in `Experiments/`
- the final model scripts in `Model/`

Differences in hardware, CUDA configuration, and package versions may lead to small differences in reproduced results.

---

---

## Acknowledgments

This repository was developed for:

**Jump AI(.py) 2025: 3rd AI Drug Discovery Competition**

and reflects our work on practical machine learning approaches for SMILES-based molecular bioactivity prediction.

---

## Contact

For questions, suggestions, or collaboration, please open an issue in this repository.
