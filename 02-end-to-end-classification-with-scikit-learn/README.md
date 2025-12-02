# Mini Project 2: End-to-End Classification with scikit-learn

## Overview
A complete binary classification pipeline using the breast cancer dataset from scikit-learn. This project demonstrates the full ML workflow: data loading, preprocessing, training, and evaluation.

## Project Goal
Build a logistic regression classifier to predict whether a tumor is malignant or benign based on 30 numerical features.

## Dataset
- **Source**: scikit-learn's `load_breast_cancer()`
- **Samples**: 569 (212 malignant, 357 benign)
- **Features**: 30 numerical features (mean radius, mean texture, etc.)
- **Task**: Binary classification

## Workflow
1. **Load and Explore**: Load data and examine distributions
2. **Train/Test Split**: 80/20 split with stratification
3. **Feature Scaling**: StandardScaler to normalize features
4. **Model Training**: Logistic Regression with default parameters
5. **Evaluation**: Accuracy, confusion matrix, classification report
6. **Visualization**: Confusion matrix heatmap

## Results
- **Test Accuracy**: 97.37%
- **Errors**: 3 out of 114 test samples
  - False Positives: 2
  - False Negatives: 1

## Key Learnings
- Importance of feature scaling for gradient-based models
- Train/test split prevents overfitting
- Confusion matrix reveals model behavior beyond accuracy
- Logistic regression performs excellently on this dataset

## Libraries Used
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Files
- `end-to-end-pipeline-with-sckikit-learn.py` - Main pipeline code
- `confusion_matrix.png` - Visualization of results
## How to Run
```bash
python end-to-end-pipeline-with-sckikit-learn.py
```

## Next Steps
- Experiment with regularization strength (C parameter)
- Try different models (Random Forest, SVM)
- Test on other datasets (Iris, Wine)
