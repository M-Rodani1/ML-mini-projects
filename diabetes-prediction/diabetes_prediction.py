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
