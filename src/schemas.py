"""Column schemas for the readmission prediction pipeline.

These schemas define the contract between data stages. Both the real MIMIC-IV
loader and the synthetic generator must produce DataFrames matching these schemas.
"""

PATIENTS_COLS = {
    "subject_id": "int64",
    "gender": "object",
    "anchor_age": "int64",
}

ADMISSIONS_COLS = {
    "subject_id": "int64",
    "hadm_id": "int64",
    "admittime": "datetime64[ns]",
    "dischtime": "datetime64[ns]",
    "admission_type": "object",
    "insurance": "object",
    "marital_status": "object",
    "race": "object",
    "hospital_expire_flag": "int64",
}

LABS_COLS = {
    "subject_id": "int64",
    "hadm_id": "int64",
    "glucose": "float64",
    "creatinine": "float64",
    "hemoglobin": "float64",
    "white_blood_cells": "float64",
    "platelets": "float64",
    "sodium": "float64",
    "potassium": "float64",
    "bicarbonate": "float64",
}

VITALS_COLS = {
    "subject_id": "int64",
    "hadm_id": "int64",
    "heart_rate": "float64",
    "systolic_bp": "float64",
    "diastolic_bp": "float64",
    "temperature": "float64",
    "respiratory_rate": "float64",
    "spo2": "float64",
}

# Final feature matrix for Stage 1
FEATURE_MATRIX_COLS = {
    "subject_id": "int64",
    "hadm_id": "int64",
    "age": "int64",
    "gender_M": "int64",
    "los_days": "float64",
    "n_prior_admissions": "int64",
    "admission_type_emergency": "int64",
    # Labs (last value during stay)
    "glucose": "float64",
    "creatinine": "float64",
    "hemoglobin": "float64",
    "white_blood_cells": "float64",
    "platelets": "float64",
    "sodium": "float64",
    "potassium": "float64",
    "bicarbonate": "float64",
    # Vitals (last value during stay)
    "heart_rate": "float64",
    "systolic_bp": "float64",
    "diastolic_bp": "float64",
    "temperature": "float64",
    "respiratory_rate": "float64",
    "spo2": "float64",
    # Target
    "readmission_30d": "int64",
}
