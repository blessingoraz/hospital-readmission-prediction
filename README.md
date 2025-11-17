# Hospital Readmission Prediction
*A Machine Learning API for Predicting 30-Day Readmission Risk Among Diabetic Patients* 

This project builds a Random Forest–based predictive model, wraps it inside a FastAPI web service, packages it using Docker, and deploys it on Render.
It provides real-time readmission probability estimates to support better post-discharge planning and reduce preventable readmissions.


## Problem Description

Hospital readmissions are a critical healthcare concern — they increase costs, strain resources, and often reflect gaps in post-discharge care.
By predicting a patient’s 30-day readmission risk before discharge, hospitals can:
- Prioritize high-risk patients
- Improve follow-up care
- Reduce strain on clinical resources
- Lower healthcare costs

This model uses demographic, clinical, and hospitalization-related features such as:
- Length of hospital stay
- Number of medications and procedures
- Previous inpatient or emergency visits
- Prescribed diabetes medications

The output is a probability of readmission and a binary risk label.

### Goal
The goal is to provide early warnings for high-risk patients so hospitals can plan targeted interventions and reduce preventable readmissions.

---

## Dataset

Source: Diabetes 130-US Hospitals dataset
Provider: UCI Machine Learning Repository
Link: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

- ~100,000 hospital encounters
- Years: 1999–2008
- Patients diagnosed with diabetes
- 50+ features (demographics, medications, lab results, diagnoses, visit counts)

**Target**: Readmission within 30 days mapped to binary
`readmitted_flag = 1 if readmitted in {"<30", ">30"} else 0`

---

## 📊 Exploratory Data Analysis (EDA)

### EDA Summary

#### Missing values: 
- Columns like weight, payer_code, medical_specialty, max_glu_serum, and A1Cresult had >50% missing values and were dropped.

#### Categorical handling: 
- All string columns were lowercased and replaced spaces with underscores.
- Categorical missing values were filled with 'NA'

#### Numerical handling
- Numerical missing values were filled with 0.0

#### Age Encoding
- Age encoding: Age bins ([50-60)) mapped to midpoints (e.g. 55).


#### Top predictive features(RandomForest):
 - number_inpatient
 - time_in_hospital
 - num_medications
 - number_emergency
 - num_lab_procedures

---

## Model Development
Primary model: RandomForestClassifier

```
max_depth = 15
min_samples_leaf = 3
class_weight = "balanced"
max_features = "sqrt"
random_state = 1
```

### Additional algorithms tested:

- Logistic Regression

- XGBoost (early trials)


### Metrics

| Model               | AUC (val)  | Precision  | Recall  | F1    | 
| ------------------- | ---------- | ---------- | ------- | ----- | 
| Logistic Regression | 0.633      | 0.571      | 0.010   | 0.021 | 
| RandomForest (v1)   | 0.633      | 0.170      | 0.505   | 0.254 | 
| XGBoost (trial)     | 0.637      | 0.786      | 0.005   | 0.010 | 

**Interpretation:**

RandomForest provides the best balance overall.

Very low recall in Logistic Regression and XGBoost → sensitive to class imbalance.

---



## API Overview

This service exposes a `POST /predict` endpoint that returns:
**Response**

```
{
  "readmitted_probability": 0.41,
  "readmitted": false,
  "threshold": 0.5
}
```

FastAPI automatically generates docs at:
👉 http://localhost:9696/docs

---

## Stack:

- FastAPI + Uvicorn for the web API

- scikit-learn RandomForestClassifier with DictVectorizer

- uv for dependency & virtualenv management

- Docker for packaging & deployment


### Project Structure
```
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

```

---

## Installation & Setup 
```
git clone https://github.com/blessingoraz/hospital-readmission-prediction.git
cd hospital-readmission-prediction

# install uv if needed
pip install uv

# install deps from pyproject + uv.lock
uv sync

# run anything inside the env:
uv run python -V
```

### Run app locally

```
# from repo root
uv run uvicorn src.predict:app --host 0.0.0.0 --port 9696
```

---

## Docker Usage

```
# build
docker build -t readmit:latest .

# run
docker run -it --rm -p 9696:9696 readmit

# test 
curl http://localhost:9696/predict

```

---

## Deployment to Render

- Push repo (with Dockerfile) to GitHub

- On Render:

  - New → Web Service → Use Docker

  - Port: 9696

  - Leave start command empty (Dockerfile ENTRYPOINT used)

  - Test https://hospital-readmission-prediction-14p1.onrender.com/predict

![alt text](https://github.com/blessingoraz/hospital-readmission-prediction/blob/main/visuals/render_deployment.png?raw=true)
---


### Test Request

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

---

## API Usage
Predict readmission rate: `POST /predict`
### Request body (subset shown; FastAPI docs show full schema):

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

### Response

```
{
  "readmitted_probability": 0.41,
  "readmitted": false
}
```


---

## Limitations

- Dataset has several sparse clinical fields
- Strong class imbalance (11% readmitted)
- Model does not incorporate temporal patterns or ICD-9 diagnosis hierarchies
- Limited to features available in the dataset

---
## Future Improvement
- Improve recall

- Add CI/CD pipeline with GitHub Actions

- Add monitoring (latency, drift, prediction stats)

- Improve metadata logging (version, parameters, training date)

---
## Acknowledgments
- Data talks community

--- 





