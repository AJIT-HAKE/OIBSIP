import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "labelled_sentiment.csv")
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

for sentiment in ["Positive", "Negative", "Neutral"]:
    reviews = df.loc[df["sentiment"] == sentiment, "text"].dropna()
    text = " ".join(reviews.astype(str))

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        stopwords=set(STOPWORDS),
        max_words=100,
        collocations=False
    ).generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{sentiment} Review WordCloud")
    plt.tight_layout()
    filename = sentiment.lower() + "_wordcloud.png"
    plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs/{filename}")

print("WordCloud analysis completed!")
