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
