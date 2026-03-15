"""
DeepCSAT Project
Step 3: Feature Engineering
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
df = pd.read_csv("data/cleaned_data.csv")

print("\nOriginal Columns:", df.columns)

# ------------------------------------------------
# 1. Convert date columns to datetime
# ------------------------------------------------

date_columns = [
    "Issue_reported at",
    "issue_responded",
    "Survey_response_Date"
]

for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# ------------------------------------------------
# 2. Extract useful time features
# ------------------------------------------------

df["Issue_Hour"] = df["Issue_reported at"].dt.hour
df["Response_Hour"] = df["issue_responded"].dt.hour
df["Survey_Hour"] = df["Survey_response_Date"].dt.hour

# Drop original date columns
df.drop(columns=date_columns, inplace=True)

# ------------------------------------------------
# 3. Encode categorical variables
# ------------------------------------------------

label_encoder = LabelEncoder()

categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# ------------------------------------------------
# 4. Save processed dataset
# ------------------------------------------------

df.to_csv("data/processed_data.csv", index=False)

print("\nFinal Shape:", df.shape)
print("\n✅ Feature engineering completed!")