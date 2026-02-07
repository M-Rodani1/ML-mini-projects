import pandas as pd
import numpy as np
df = pd.read_csv('quikr_car.csv')

print("data types before cleaning")
print(df.info())

print("\n--- Unique Values Inspection ---")
print("First 5 Prices:", df['Price'].unique()[:5])
print("First 5 Kms:   ", df['kms_driven'].unique()[:5])

ask_price_count = (df['Price'] == 'Ask For Price').sum()
print(f"\nRows with 'Ask For Price': {ask_price_count}")


# Clean 'Price'
df = df[df['Price'] != "Ask For Price"]
df['Price'] = df['Price'].str.replace(',', '').astype(int)

# Clean 'kms_driven'
# First, remove the " kms" text and commas
df['kms_driven'] = df['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')

# Remove rows where kms_driven is missing (NaN)
df = df[df['kms_driven'].notna()]

# Now checking .isnumeric() won't crash because there are no NaNs left
df = df[df['kms_driven'].str.isnumeric()]
df['kms_driven'] = df['kms_driven'].astype(int)

# Clean 'fuel_type'
df = df[~df['fuel_type'].isna()]

# Clean 'year'
df = df[df['year'].str.isnumeric()]
df['year'] = df['year'].astype(int)

# Clean 'name'
df['name'] = df['name'].str.split(' ').str.slice(0, 3).str.join(' ')

# Reset Index
df = df.reset_index(drop=True)

print("\nCleaning Verification")
print(df.info())
print(f"Remaining Rows: {len(df)}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

#  Define X and y
X = df.drop(columns=['Price'])
y = df['Price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=433)

one = OneHotEncoder()

one.fit(X[['name', 'company', 'fuel_type']])

#  Create Column Transformer
column_trans = make_column_transformer(
    (OneHotEncoder(categories=one.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# Pipeline
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)

#  Train
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"R2 Score: {r2:.4f}")
