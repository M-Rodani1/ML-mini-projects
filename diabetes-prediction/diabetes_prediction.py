import pandas as pd
import numpy as np
df = pd.read_csv('diabetes.csv')

critical_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
print('checking for invalid zeros')
zeros_count = (df[critical_cols]==0).sum()
print('zeros count: ',zeros_count)

df[critical_cols] = (df[critical_cols].replace(0,np.nan))
print('missing values after marking')
print(df.isnull().sum())

from sklearn.model_selection import train_test_split
X = df.drop('Outcome',axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
print('split complete')
print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape:     {X_test.shape}")

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

print(f"Missing values in Train: {np.isnan(X_train_imputed).sum()}")
print(f"Missing values in Test:  {np.isnan(X_test_imputed).sum()}")

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train_imputed)
X_test_scaled = sc.transform(X_test_imputed)

print(f"Mean of first feature (Train): {X_train_scaled[:, 0].mean():.4f}")
print(f"Std of first feature (Train):  {X_train_scaled[:, 0].std():.4f}")

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix)
