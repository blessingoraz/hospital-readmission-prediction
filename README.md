# Hospital Readmission Prediction
A machine learning API that predicts whether a hospitalized diabetic patient is likely to be readmitted within 30 days.
Built with Random Forest, served via FastAPI, containerized with Docker, and deployed to Render.

# Problem Description

Hospital readmissions are a critical healthcare concern — they increase costs, strain resources, and often reflect gaps in post-discharge care.
This project predicts the likelihood of readmission for diabetic patients based on demographic and clinical features such as:

- Length of hospital stay

- Number of medications and procedures

- Previous inpatient or emergency visits

- Prescribed diabetes medications

The goal is to provide early warnings for high-risk patients so hospitals can plan targeted interventions and reduce preventable readmissions.

# 📊 Exploratory Data Analysis (EDA)

The dataset used is the Diabetes 130-US Hospitals dataset from the UCI Machine Learning Repository, containing over 100,000 hospital encounters (1999–2008).

## EDA Summary

- Missing values: Columns like weight, payer_code, medical_specialty, max_glu_serum, and A1Cresult had >50% missing values and were dropped.

- Categorical handling: All string columns were lowercased and replaced spaces with underscores.

- Numerical missing values: Filled with 0.0; categorical with 'NA'.

- Age encoding: Age bins ([50-60)) mapped to midpoints (e.g. 55).

- Target variable: Converted readmitted → 1 if <30 else 0.

- Readmission rate: ~11% (imbalanced dataset).

- Feature importance (RandomForest):
Top predictive features:
1. number_inpatient
2. time_in_hospital
3. num_medications
4. number_emergency
5. num_lab_procedures

# Overview

This service exposes a POST /predict endpoint that returns:

readmitted_probability: model probability (0–1)

readmitted: boolean flag using a configurable threshold (default 0.5)

# Stack:

- FastAPI + Uvicorn for the web API

- scikit-learn RandomForestClassifier with DictVectorizer

- uv for dependency & virtualenv management

- Docker for packaging & deployment

# Project Structure
.
├─ pyproject.toml
├─ uv.lock
├─ Dockerfile
├─ data/
│  ├─ raw/diabetic_data.csv
│  └─ interim/               # (optional) saved splits
├─ models/
│  ├─ model.bin              # trained pipeline (DictVectorizer + RF)
│  └─ metadata.json          # model info (optional)
├─ src/
│  ├─ train.py               # trains and saves model.bin
│  └─ predict.py             # FastAPI app (POST /predict)
└─ README.md

# Data

Dataset: Diabetes 130-US hospitals for years 1999–2008 (UCI)

Target: Readmission within 30 days mapped to binary
readmitted_flag = 1 if readmitted in {"<30", ">30"} else 0 (or your mapping if different)

Basic cleaning:

Replace '?' → NaN

Drop sparse columns (e.g., weight, payer_code, medical_specialty, max_glu_serum, A1Cresult)

Lowercase + underscore categorical values

Fill categorical NA, numeric 0.0

Map age bins to mid-points (e.g., [60-70) → 65)

# Model
RandomForestClassifier

max_depth=15, min_samples_leaf=3, class_weight='balanced', max_features='sqrt', random_state=1

DictVectorizer handles one-hot encoding inside a Pipeline

Saved as models/model.bin via pickle

# Quickstart

## Setup 
```
git clone https://github.com/blessingoraz/hospital-readmission-prediction.git
cd readmission-prediction

# install uv if needed
pip install uv

# install deps from pyproject + uv.lock
uv sync
# run anything inside the env:
uv run python -V
```

## Run app locally

```
# from repo root
uv run uvicorn src.predict:app --host 0.0.0.0 --port 9696
# open docs
# http://localhost:9696/docs

```

## Test Request

```

curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "time_in_hospital": 5,
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 12,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 1,
    "number_diagnoses": 7,
    "age": "[60-70)",
    "race": "Caucasian",
    "gender": "Female",
    "metformin": "No",
    "insulin": "Steady"
  }'
```

# Docker

```

# build
docker build -t readmit:latest .
# run
docker run -it --rm -p 9696:9696 readmit
# test (same curl as above)

```

# API
`POST /predict``
Request body (subset shown; FastAPI docs show full schema):

```
{
  "time_in_hospital": 5,
  "num_lab_procedures": 45,
  "num_procedures": 1,
  "num_medications": 12,
  "number_outpatient": 0,
  "number_emergency": 0,
  "number_inpatient": 1,
  "number_diagnoses": 7,
  "age": "[60-70)",
  "race": "Caucasian",
  "gender": "Female",
  "metformin": "No",
  "insulin": "Steady"
}
```

# Response
```
{
  "readmitted_probability": 0.41,
  "readmitted": false,
  "threshold": 0.5
}
```

# Deployment

- Push repo (with Dockerfile) to GitHub

- On Render:

  - New → Web Service → Use Docker

  - Port: 9696

  - Leave start command empty (Dockerfile ENTRYPOINT used)

  - Test https://hospital-readmission-prediction-14p1.onrender.com/predict


# Metrics

| Model               | AUC (val) | Precision | Recall | F1  | Notes                 |
| ------------------- | --------- | --------- | ------ | --- | --------------------- |
| Logistic Regression | 0.63      | 0.53      | 0.02   | .03 | baseline              |
| RandomForest (v1)   | 0.64      | 0.50      | 0.01   | .02 | class_weight balanced |
| XGBoost (trial)     | 0.66      | —         | —      | —   | coming later          |





