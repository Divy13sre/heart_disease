# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 2: Load Dataset from Local Path (Windows-compatible path)
df = pd.read_csv(r"C:\Users\DELL\Downloads\heart.csv")

# Step 3: Handle Missing Values
df.fillna(df.median(numeric_only=True), inplace=True)

# Step 4: Drop Duplicates
df.drop_duplicates(inplace=True)

# Step 5: One-Hot Encode Categorical Features
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Step 6: Normalize Numerical Features (excluding target)
target = 'HeartDisease'
numerical_cols = df.select_dtypes(include='number').columns.tolist()
if target in numerical_cols:
    numerical_cols.remove(target)

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Step 7: Remove Outliers using IQR
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# Step 8: Save the Cleaned Dataset
df.to_csv(r"C:\Users\DELL\Downloads\heart_cleaned.csv", index=False)
print(" Cleaned dataset saved as 'heart_cleaned.csv'")

