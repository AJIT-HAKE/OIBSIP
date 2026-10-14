import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Task 3: Data Cleaning – Titanic Dataset

**Internship:** Oasis Infobyte Data Analytics  
**Objective:** Identify missing values, standardise text, correct data types, handle missing data, inspect outliers, and export a clean dataset.

> The source is the public Titanic dataset. Missing values are handled using documented rules. Outliers are reported, not automatically deleted."""))

cells.append(nbf.v4.new_markdown_cell("## 1. Import Libraries"))
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)
df = pd.read_csv("data/titanic_messy.csv")
print("Dataset shape:", df.shape)
display(df.head())"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Initial Data Quality Check"))
cells.append(nbf.v4.new_code_cell("""before = df.copy()
quality = pd.DataFrame({
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isna().sum(),
    "Missing Percent": (df.isna().mean() * 100).round(2),
    "Unique Values": df.nunique()
})
display(quality)
print("Exact duplicate rows:", df.duplicated().sum())
print("Total missing cells:", df.isna().sum().sum())"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Remove Duplicates and Standardise Text"))
cells.append(nbf.v4.new_code_cell("""df = df.drop_duplicates().copy()

df["Sex"] = df["Sex"].astype("string").str.strip().str.lower()
df["Sex"] = df["Sex"].replace({"m": "male", "f": "female"})
df["Sex"] = df["Sex"].map({"male": "Male", "female": "Female"})

for col in ["Name", "Ticket", "Cabin"]:
    df[col] = df[col].astype("string").str.strip()

df["Embarked"] = df["Embarked"].astype("string").str.strip().str.upper()
print("Text standardisation completed.")"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Correct Data Types"))
cells.append(nbf.v4.new_code_cell("""df["PassengerId"] = df["PassengerId"].astype("string")
for col in ["Survived", "Pclass", "SibSp", "Parch"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
for col in ["Age", "Fare"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

display(df.dtypes.to_frame("Data Type"))"""))

cells.append(nbf.v4.new_markdown_cell("""## 5. Handle Missing Values

- **Age:** Fill with the overall median.
- **Embarked:** Fill with the most frequent category.
- **Fare:** Fill using the median fare within passenger class, then the overall median.
- **Cabin:** Use `Unknown` to preserve the fact that the cabin is not recorded; this does not recover the real cabin number."""))
cells.append(nbf.v4.new_code_cell("""df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])
df["Fare"] = df.groupby("Pclass")["Fare"].transform(
    lambda s: s.fillna(s.median())
)
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Cabin"] = df["Cabin"].fillna("Unknown")

print("Missing cells after cleaning:", df.isna().sum().sum())
display(df.isna().sum().to_frame("Missing Values"))"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Outlier Detection Using IQR"))
cells.append(nbf.v4.new_code_cell("""outlier_rows = []
for col in ["Age", "Fare", "SibSp", "Parch"]:
    s = pd.to_numeric(df[col], errors="coerce")
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((s < low) | (s > high)).sum())
    outlier_rows.append({
        "Column": col, "Lower Bound": low, "Upper Bound": high,
        "Outlier Count": count,
        "Decision": "Retain; values may be genuine"
    })
outliers = pd.DataFrame(outlier_rows)
display(outliers)
outliers.to_csv("outputs/outlier_report.csv", index=False)

df["Fare"].plot(kind="box", title="Fare Outlier Inspection")
plt.ylabel("Fare")
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 7. Before-and-After Comparison"))
cells.append(nbf.v4.new_code_cell("""summary = pd.DataFrame([
    {"Stage": "Before Cleaning", "Rows": len(before),
     "Missing Cells": int(before.isna().sum().sum()),
     "Duplicate Rows": int(before.duplicated().sum())},
    {"Stage": "After Cleaning", "Rows": len(df),
     "Missing Cells": int(df.isna().sum().sum()),
     "Duplicate Rows": int(df.duplicated().sum())}
])
display(summary)
summary.to_csv("outputs/before_after_summary.csv", index=False)

missing = before.isna().sum()
missing = missing[missing > 0].sort_values()
missing.plot(kind="barh", title="Missing Values Before Cleaning",
             xlabel="Missing Count")
plt.tight_layout()
plt.show()

plt.bar(["Before Cleaning", "After Cleaning"],
        [before.isna().sum().sum(), df.isna().sum().sum()])
plt.title("Missing Cells: Before vs After")
plt.ylabel("Missing Cell Count")
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 8. Export Cleaned Dataset and Final Report"))
cells.append(nbf.v4.new_code_cell("""df.to_csv("outputs/titanic_cleaned.csv", index=False)
quality.to_csv("outputs/data_quality_report.csv")
pd.DataFrame({
    "Column": df.columns,
    "Final Data Type": [str(df[c].dtype) for c in df.columns],
    "Missing Values": [int(df[c].isna().sum()) for c in df.columns]
}).to_csv("outputs/final_data_types.csv", index=False)

print("Cleaned data and reports saved in outputs/.")
print("Final shape:", df.shape)
print("Remaining missing cells:", df.isna().sum().sum())"""))

cells.append(nbf.v4.new_markdown_cell("""## 9. Conclusion

The dataset was inspected and cleaned by removing exact duplicates, standardising text, converting selected columns to suitable types, handling missing values, and reporting numeric outliers. The cleaned CSV and quality reports are saved in `outputs/`.

**Limitation:** The source dataset has no exact duplicate rows. Outliers are retained because extreme values can be genuine passenger records. `Cabin = Unknown` means the cabin information was missing in the source."""))

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, "Data_Cleaning_Titanic.ipynb")
print("Notebook created: Data_Cleaning_Titanic.ipynb")
