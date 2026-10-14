import pandas as pd

file_path = "data/Online Retail.xlsx"
df = pd.read_excel(file_path)

print("ORIGINAL DATASET:", df.shape)

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

# Remove rows without CustomerID
df = df.dropna(subset=["CustomerID"]).copy()

# Convert data types
df["CustomerID"] = df["CustomerID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Calculate transaction value
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# Keep valid purchase transactions
# Exclude cancellations, returns, and non-positive values for RFM analysis
df = df[
    (~df["InvoiceNo"].astype(str).str.startswith("C"))
    & (df["Quantity"] > 0)
    & (df["UnitPrice"] > 0)
].copy()

print("\nCLEANED DATASET:", df.shape)
print("\nMISSING VALUES AFTER CLEANING:")
print(df.isnull().sum().to_string())

print("\nDATE RANGE:")
print(df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())

print("\nUNIQUE CUSTOMERS:", df["CustomerID"].nunique())
print("UNIQUE INVOICES:", df["InvoiceNo"].nunique())
print("TOTAL SALES VALUE:", round(df["TotalAmount"].sum(), 2))
print("AVERAGE TRANSACTION VALUE:", round(df["TotalAmount"].mean(), 2))

import os
os.makedirs("outputs", exist_ok=True)

df.to_csv("outputs/cleaned_retail_data.csv", index=False)

summary = pd.DataFrame({
    "Metric": [
        "Original rows",
        "Rows after removing duplicates and missing CustomerID",
        "Rows after excluding cancellations and invalid purchases",
        "Unique customers",
        "Unique invoices",
        "Total sales value",
        "Average transaction value"
    ],
    "Value": [
        pd.read_excel(file_path).shape[0],
        "See cleaned processing steps",
        len(df),
        df["CustomerID"].nunique(),
        df["InvoiceNo"].nunique(),
        round(df["TotalAmount"].sum(), 2),
        round(df["TotalAmount"].mean(), 2)
    ]
})
summary.to_csv("outputs/data_cleaning_summary.csv", index=False)
print("\nSaved: outputs/cleaned_retail_data.csv")
print("Saved: outputs/data_cleaning_summary.csv")
