import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

before = pd.read_csv("data/titanic_messy.csv")
after = pd.read_csv("outputs/titanic_cleaned.csv")

# Missing values before cleaning
missing = before.isna().sum()
missing = missing[missing > 0].sort_values()

plt.figure(figsize=(9, 5))
missing.plot(kind="barh")
plt.title("Missing Values Before Cleaning")
plt.xlabel("Number of Missing Values")
plt.tight_layout()
plt.savefig("outputs/missing_values_before.png", dpi=150)
plt.close()

# Before vs after missing cells
comparison = [
    before.isna().sum().sum(),
    after.isna().sum().sum()
]
plt.figure(figsize=(7, 5))
plt.bar(["Before Cleaning", "After Cleaning"], comparison)
plt.title("Missing Values: Before vs After")
plt.ylabel("Number of Missing Cells")
plt.tight_layout()
plt.savefig("outputs/missing_values_comparison.png", dpi=150)
plt.close()

# Age distribution after cleaning
plt.figure(figsize=(8, 5))
after["Age"].dropna().hist(bins=25)
plt.title("Passenger Age Distribution After Cleaning")
plt.xlabel("Age")
plt.ylabel("Passenger Count")
plt.tight_layout()
plt.savefig("outputs/age_distribution.png", dpi=150)
plt.close()

# Fare outliers visualisation
plt.figure(figsize=(8, 4))
after["Fare"].dropna().plot(kind="box")
plt.title("Fare Outlier Inspection")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("outputs/fare_boxplot.png", dpi=150)
plt.close()

print("Visualizations created successfully!")
print("Check outputs/ for four PNG files.")
