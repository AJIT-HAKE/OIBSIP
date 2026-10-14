import os
import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)

# 1. Load original dataset
df = pd.read_csv("data/titanic_messy.csv")
before = df.copy()

print("Original shape:", df.shape)

# 2. Create initial data quality report
quality = pd.DataFrame({
    "Column": df.columns,
    "Data_Type_Before": [str(df[c].dtype) for c in df.columns],
    "Missing_Count": [df[c].isna().sum() for c in df.columns],
    "Missing_Percent": [(df[c].isna().mean() * 100).round(2) for c in df.columns],
    "Unique_Values": [df[c].nunique(dropna=True) for c in df.columns]
})
quality.to_csv("outputs/data_quality_report.csv", index=False)

duplicate_count = int(df.duplicated().sum())

# 3. Remove exact duplicate rows
df = df.drop_duplicates().copy()

# 4. Standardise text formatting
df["Sex"] = df["Sex"].astype("string").str.strip().str.lower()
df["Sex"] = df["Sex"].replace({
    "m": "male", "male": "male",
    "f": "female", "female": "female"
}).map({"male": "Male", "female": "Female"})

df["Embarked"] = df["Embarked"].astype("string").str.strip().str.upper()
df["Name"] = df["Name"].astype("string").str.strip()
df["Ticket"] = df["Ticket"].astype("string").str.strip()
df["Cabin"] = df["Cabin"].astype("string").str.strip()

# 5. Correct data types
df["PassengerId"] = df["PassengerId"].astype("string")
df["Ticket"] = df["Ticket"].astype("string")
for col in ["Survived", "Pclass", "SibSp", "Parch"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
for col in ["Age", "Fare"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 6. Handle missing values with documented strategies
# Age: median is robust to skew and extreme values.
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked: mode is suitable for a small number of missing categories.
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])

# Fare: median by passenger class, then overall median if needed.
df["Fare"] = df.groupby("Pclass")["Fare"].transform(
    lambda s: s.fillna(s.median())
)
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Cabin has many missing values; retain the missingness information.
df["Cabin"] = df["Cabin"].fillna("Unknown")

# 7. Detect numeric outliers using IQR; report them but retain valid observations.
outlier_rows = []
for col in ["Age", "Fare", "SibSp", "Parch"]:
    series = pd.to_numeric(df[col], errors="coerce")
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    outlier_rows.append({
        "Column": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower,
        "Upper_Bound": upper,
        "Outlier_Count": count,
        "Decision": "Retained for analysis; extreme values may be genuine"
    })
pd.DataFrame(outlier_rows).to_csv("outputs/outlier_report.csv", index=False)

# 8. Before vs after summary
summary = pd.DataFrame([
    {
        "Stage": "Before Cleaning",
        "Rows": len(before),
        "Columns": len(before.columns),
        "Missing_Cells": int(before.isna().sum().sum()),
        "Duplicate_Rows": int(before.duplicated().sum()),
        "Missing_Age": int(before["Age"].isna().sum()),
        "Missing_Fare": int(before["Fare"].isna().sum()),
        "Missing_Embarked": int(before["Embarked"].isna().sum()),
        "Missing_Cabin": int(before["Cabin"].isna().sum())
    },
    {
        "Stage": "After Cleaning",
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing_Cells": int(df.isna().sum().sum()),
        "Duplicate_Rows": int(df.duplicated().sum()),
        "Missing_Age": int(df["Age"].isna().sum()),
        "Missing_Fare": int(df["Fare"].isna().sum()),
        "Missing_Embarked": int(df["Embarked"].isna().sum()),
        "Missing_Cabin": int(df["Cabin"].isna().sum())
    }
])
summary.to_csv("outputs/before_after_summary.csv", index=False)

# 9. Save cleaned dataset and dtype report
df.to_csv("outputs/titanic_cleaned.csv", index=False)
pd.DataFrame({
    "Column": df.columns,
    "Final_Data_Type": [str(df[c].dtype) for c in df.columns],
    "Missing_Values": [int(df[c].isna().sum()) for c in df.columns]
}).to_csv("outputs/final_data_types.csv", index=False)

print("\nCleaning completed!")
print("Rows before:", len(before))
print("Rows after:", len(df))
print("Duplicate rows removed:", duplicate_count)
print("Missing cells before:", int(before.isna().sum().sum()))
print("Missing cells after:", int(df.isna().sum().sum()))
print("\nBefore vs After:")
print(summary.to_string(index=False))
print("\nFiles saved in outputs/")
