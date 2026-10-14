import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("outputs", exist_ok=True)

# 1. Load and inspect dataset
df = pd.read_csv("data/train.csv")
print("Original shape:", df.shape)
print("Missing values before cleaning:", df.isna().sum().sum())

# 2. Clean data
df = df.drop_duplicates().copy()
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()
df = df[df["medv"] > 0]

df.to_csv("outputs/cleaned_housing_data.csv", index=False)
print("Cleaned shape:", df.shape)

# 3. Save data quality report
quality = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [str(df[c].dtype) for c in df.columns],
    "Missing_Values": [int(df[c].isna().sum()) for c in df.columns],
    "Unique_Values": [int(df[c].nunique()) for c in df.columns]
})
quality.to_csv("outputs/data_quality_report.csv", index=False)

# 4. Exploratory Data Analysis
df.hist(figsize=(14, 10), bins=20)
plt.suptitle("Housing Dataset - Feature Distributions")
plt.tight_layout()
plt.savefig("outputs/feature_distributions.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["rm"], df["medv"], alpha=0.65)
plt.xlabel("Average Number of Rooms (rm)")
plt.ylabel("Median House Value (medv)")
plt.title("Rooms vs House Value")
plt.tight_layout()
plt.savefig("outputs/rooms_vs_price.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png", dpi=150)
plt.close()

# 5. Split features and target
X = df.drop(columns=["medv"])
y = df["medv"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

metrics = pd.DataFrame([{
    "Model": "Linear Regression",
    "Training_Rows": len(X_train),
    "Testing_Rows": len(X_test),
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2_Score": r2
}])
metrics.to_csv("outputs/model_metrics.csv", index=False)

comparison = pd.DataFrame({
    "Actual_Value": y_test.values,
    "Predicted_Value": predictions.round(2)
})
comparison.to_csv("outputs/predictions.csv", index=False)

plt.figure(figsize=(7, 6))
plt.scatter(y_test, predictions, alpha=0.7)
low = min(y_test.min(), predictions.min())
high = max(y_test.max(), predictions.max())
plt.plot([low, high], [low, high], linestyle="--")
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted Values")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
plt.close()

print("\nModel Evaluation:")
print(metrics.to_string(index=False))
print("\nExample predictions:")
print(comparison.head(10).to_string(index=False))
print("\nProject completed! Check the outputs/ folder.")
