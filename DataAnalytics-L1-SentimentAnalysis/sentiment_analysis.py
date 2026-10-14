import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_DIR / "sentiment_dataset.csv")

print("Original dataset shape:", df.shape)
print("\nMissing values:\n", df[["text", "stars"]].isnull().sum())
print("\nStar ratings:\n", df["stars"].value_counts().sort_index())

# Clean required columns
df = df.dropna(subset=["text", "stars"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df = df.drop_duplicates(subset=["text", "stars"]).copy()

# Convert star ratings to sentiment labels
def assign_sentiment(stars):
    if stars <= 2:
        return "Negative"
    elif stars == 3:
        return "Neutral"
    return "Positive"

df["sentiment"] = df["stars"].apply(assign_sentiment)

# Save labelled data
df[["text", "stars", "sentiment"]].to_csv(
    DATA_DIR / "labelled_sentiment.csv", index=False
)

print("\nCleaned dataset shape:", df.shape)
print("\nSentiment distribution:\n", df["sentiment"].value_counts())

# Plot sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="sentiment", order=["Negative", "Neutral", "Positive"])
plt.title("Distribution of Yelp Review Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sentiment_distribution.png", dpi=150)
plt.close()

print("\nSaved labelled dataset and sentiment distribution chart.")
