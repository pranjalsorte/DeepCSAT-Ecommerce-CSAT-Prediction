"""
DeepCSAT Project
Dataset Inspection Script
"""

import pandas as pd

# Load dataset
df = pd.read_csv("data/eCommerce_Customer_support_data.csv")

print("\n✅ Dataset Loaded Successfully")

# Shape
print("\n📊 Dataset Shape:")
print(df.shape)

# Column names
print("\n📌 Columns:")
print(df.columns.tolist())

# Data types
print("\n🧾 Data Types:")
print(df.dtypes)

# Missing values
print("\n❓ Missing Values:")
print(df.isnull().sum())

# Preview
print("\n👀 First 5 Rows:")
print(df.head())