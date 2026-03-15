"""
DeepCSAT Project
Step 2: Data Cleaning
"""

import pandas as pd

# Load dataset
df = pd.read_csv("data/eCommerce_Customer_support_data.csv")

print("\nOriginal Shape:", df.shape)

# ---------------------------------------------------
# 1. Drop columns with extremely high missing values
# ---------------------------------------------------

drop_columns = [
    "connected_handling_time",  # ~99% missing
    "order_date_time",          # mostly missing
]

df.drop(columns=drop_columns, inplace=True)

# ---------------------------------------------------
# 2. Handle Missing Values
# ---------------------------------------------------

# Text column → replace with 'No Remark'
df["Customer Remarks"] = df["Customer Remarks"].fillna("No Remark")

# Categorical columns → replace with 'Unknown'
categorical_cols = [
    "Customer_City",
    "Product_category",
    "Order_id"
]

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# Numerical column → median fill
df["Item_price"] = df["Item_price"].fillna(df["Item_price"].median())

# ---------------------------------------------------
# 3. Remove duplicates
# ---------------------------------------------------

df.drop_duplicates(inplace=True)

print("\nCleaned Shape:", df.shape)

# ---------------------------------------------------
# 4. Save cleaned dataset
# ---------------------------------------------------

df.to_csv("data/cleaned_data.csv", index=False)

print("\n✅ Cleaned dataset saved successfully!")