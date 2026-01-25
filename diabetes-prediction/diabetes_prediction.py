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