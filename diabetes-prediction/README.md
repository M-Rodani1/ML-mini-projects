# Diabetes Prediction

A machine learning classifier to predict diabetes diagnosis using the Pima Diabetes Dataset.

## Methodology

**Data Preprocessing:**
- Identified and handled invalid zero values in critical medical features (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
- Replaced zeros with NaN and applied median imputation
- Standardised features using StandardScaler

**Model:**
- Support Vector Machine (SVM) with RBF kernel
- Train-test split: 80/20 with stratification to preserve class distribution

## Features

The model uses 8 medical predictors:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## Usage

```bash
python diabetes_prediction.py
```

## Dataset

Pima Diabetes Database - 768 samples with binary classification (diabetic/non-diabetic)
