import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "labelled_sentiment.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA).dropna(subset=["text", "sentiment"])
X = df["text"].astype(str)
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = []
all_predictions = {}

for name, classifier in models.items():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

    print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    all_predictions[name] = predictions

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Weighted Precision": precision,
        "Weighted Recall": recall,
        "Weighted F1": f1
    })

    print(classification_report(y_test, predictions, zero_division=0))

    cm = confusion_matrix(
        y_test, predictions,
        labels=["Negative", "Neutral", "Positive"]
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"]
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    filename = name.lower().replace(" ", "_")
    plt.savefig(OUT / f"{filename}_confusion_matrix.png", dpi=150)
    plt.close()

# Save and compare metrics
metrics = pd.DataFrame(results)
metrics.to_csv(OUT / "model_comparison.csv", index=False)
print("\nMODEL COMPARISON")
print(metrics.to_string(index=False))

# Save five Logistic Regression errors for review
errors = pd.DataFrame({
    "text": X_test.reset_index(drop=True),
    "actual": y_test.reset_index(drop=True),
    "predicted": pd.Series(all_predictions["Logistic Regression"])
})
errors = errors[errors["actual"] != errors["predicted"]].head(5)
errors.to_csv(OUT / "misclassified_examples.csv", index=False)
print("\nFive misclassified examples saved to outputs/misclassified_examples.csv")
print(errors.to_string(index=False))
