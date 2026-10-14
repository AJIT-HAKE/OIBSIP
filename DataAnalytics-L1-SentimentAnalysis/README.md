# Level 1 Task 4: Sentiment Analysis

## Objective
Analyze Yelp customer reviews and classify them as Positive, Negative, or Neutral using star ratings. Train and compare machine learning models for sentiment classification.

## Dataset
The project uses the Yelp review dataset containing 10,000 original reviews.

Sentiment labels:
- 1–2 stars: Negative
- 3 stars: Neutral
- 4–5 stars: Positive

After cleaning, 9,998 reviews were retained.

## Sentiment Distribution
- Positive: 6,862
- Negative: 1,675
- Neutral: 1,461

## Models
TF-IDF features with unigram and bigram text features were used with:
1. Multinomial Naive Bayes
2. Logistic Regression

## Model Results

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Naive Bayes | 69.05% | 56.82% |
| Logistic Regression | 78.30% | 74.04% |

Logistic Regression performed better on this test split. Neutral sentiment remains more difficult to classify.

## Visualizations
- Sentiment distribution
- Positive review WordCloud
- Negative review WordCloud
- Neutral review WordCloud
- Confusion matrices for both models

## Project Files
- `sentiment_analysis.py`: Data cleaning and sentiment labelling
- `train_models.py`: Model training and evaluation
- `wordcloud_analysis.py`: WordCloud generation
- `data/`: Input and labelled datasets
- `outputs/`: Charts, metrics, and misclassified examples

## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
python sentiment_analysis.py
python train_models.py
python wordcloud_analysis.py
cd ~/Desktop/OIBSIP/DataAnalytics-L1-SentimentAnalysis

cat > README.md <<'EOF'
# Level 1 Task 4: Sentiment Analysis

## Objective
Analyze Yelp reviews and classify them as Positive, Negative, or Neutral using star ratings and machine learning.

## Dataset
Yelp review dataset containing 10,000 reviews.
Source: https://raw.githubusercontent.com/justmarkham/DAT8/master/data/yelp.csv

## Sentiment Labels
- 1–2 stars: Negative
- 3 stars: Neutral
- 4–5 stars: Positive

## Workflow
1. Load and clean review data.
2. Assign sentiment labels using star ratings.
3. Explore sentiment distribution.
4. Convert review text into TF-IDF features.
5. Train and evaluate Naive Bayes and Logistic Regression.
6. Generate confusion matrices and WordCloud visualizations.

## Results
- Cleaned reviews: 9,998
- Positive: 6,862
- Negative: 1,675
- Neutral: 1,461
- Naive Bayes accuracy: 69.05%
- Logistic Regression accuracy: 78.30%

Logistic Regression performed better on this test split. Neutral sentiment was more difficult to classify.

## Project Files
- sentiment_analysis.py: Data cleaning and sentiment labelling
- train_models.py: Model training and evaluation
- wordcloud_analysis.py: WordCloud generation
- data/: Dataset files
- outputs/: Charts and evaluation results

## How to Run
Install dependencies:

    python -m pip install -r requirements.txt

Download the dataset:

    mkdir -p data outputs
    curl -L "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/yelp.csv" -o data/sentiment_dataset.csv

Run the scripts:

    python sentiment_analysis.py
    python train_models.py
    python wordcloud_analysis.py
