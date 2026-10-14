import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("""# Level 1 Task 4: Sentiment Analysis
## Oasis Infobyte Data Analytics Internship

**Objective:** Analyze Yelp reviews and classify them as Positive, Negative, or Neutral.

### Sentiment rules
- 1–2 stars: Negative
- 3 stars: Neutral
- 4–5 stars: Positive

We compare Multinomial Naive Bayes and Logistic Regression using TF-IDF text features."""),
    nbf.v4.new_markdown_cell("## 1. Load and clean the dataset"),
    nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "sentiment_dataset.csv")
print("Original shape:", df.shape)
display(df.head())
print("\\\\nMissing values:")
display(df[["text", "stars"]].isnull().sum())"""),
    nbf.v4.new_code_cell("""df = df.dropna(subset=["text", "stars"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df = df.drop_duplicates(subset=["text", "stars"]).copy()

def assign_sentiment(stars):
    if stars <= 2:
        return "Negative"
    elif stars == 3:
        return "Neutral"
    return "Positive"

df["sentiment"] = df["stars"].apply(assign_sentiment)
print("Cleaned shape:", df.shape)
display(df["sentiment"].value_counts())"""),
    nbf.v4.new_markdown_cell("## 2. Explore sentiment distribution"),
    nbf.v4.new_code_cell("""order = ["Negative", "Neutral", "Positive"]
df["sentiment"].value_counts().reindex(order).plot(
    kind="bar", figsize=(8, 5), title="Yelp Review Sentiment Distribution"
)
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()"""),
    nbf.v4.new_markdown_cell("## 3. Train and compare machine learning models"),
    nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2,
    random_state=42, stratify=df["sentiment"]
)

models = {
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
        ("model", MultinomialNB())
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
    results.append({
        "Model":
ls -lh Sentiment_Analysis.ipynb
rsync -av --exclude 'data/' --exclude '__pycache__/' --exclude '.ipynb_checkpoints/' --exclude 'outputs/misclassified_examples.csv' ~/Desktop/OIBSIP/DataAnalytics-L1-SentimentAnalysis/ ~/Desktop/OIBSIP-GitHub/DataAnalytics-L1-SentimentAnalysis/

cd ~/Desktop/OIBSIP-GitHub
git status
cd ~/Desktop/OIBSIP/DataAnalytics-L1-SentimentAnalysis
ls -lh Sentiment_Analysis.ipynb
python - <<'PY'
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("""# Level 1 Task 4: Sentiment Analysis

## Objective
Analyze Yelp reviews and classify them as Positive, Negative, or Neutral.

## Dataset
10,000 Yelp reviews. Sentiment labels are derived from star ratings:
- 1–2 stars: Negative
- 3 stars: Neutral
- 4–5 stars: Positive

## Workflow
Data cleaning, exploratory analysis, TF-IDF text features, Naive Bayes and Logistic Regression, model evaluation, and WordCloud visualizations."""),
    nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

DATA = Path("data")
OUT = Path("outputs")
print("Run this notebook from the project folder:", Path.cwd())"""),
    nbf.v4.new_markdown_cell("## 1. Load and label the data"),
    nbf.v4.new_code_cell("""df = pd.read_csv(DATA / "sentiment_dataset.csv")
df = df.dropna(subset=["text", "stars"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].drop_duplicates(subset=["text", "stars"]).copy()
df["sentiment"] = df["stars"].apply(
    lambda s: "Negative" if s <= 2 else ("Neutral" if s == 3 else "Positive")
)
print("Cleaned shape:", df.shape)
display(df[["stars", "sentiment", "text"]].head())
display(df["sentiment"].value_counts())"""),
    nbf.v4.new_markdown_cell("## 2. Sentiment distribution"),
    nbf.v4.new_code_cell("""display(Image(filename=str(OUT / "sentiment_distribution.png")))"""),
    nbf.v4.new_markdown_cell("## 3. Compare machine learning models"),
    nbf.v4.new_code_cell("""metrics = pd.read_csv(OUT / "model_comparison.csv")
display(metrics)
print("Logistic Regression performed better on the recorded test split.")"""),
    nbf.v4.new_markdown_cell("## 4. Confusion matrices"),
    nbf.v4.new_code_cell("""for filename in ["naive_bayes_confusion_matrix.png",
                 "logistic_regression_confusion_matrix.png"]:
    print(filename)
    display(Image(filename=str(OUT / filename)))"""),
    nbf.v4.new_markdown_cell("## 5. WordCloud visualizations"),
    nbf.v4.new_code_cell("""for filename in ["positive_wordcloud.png",
                 "negative_wordcloud.png", "neutral_wordcloud.png"]:
    print(filename)
    display(Image(filename=str(OUT / filename)))"""),
    nbf.v4.new_markdown_cell("""## Conclusion
Logistic Regression achieved 78.30% accuracy and a weighted F1 score of 74.04%, outperforming Naive Bayes on this test split. Neutral sentiment was harder to classify. Labels are based on star ratings, so they may not always reflect the full meaning of a review.""")
]
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
}
nbf.write(nb, "Sentiment_Analysis.ipynb")
print("Created Sentiment_Analysis.ipynb")
