#--------------------------------
#03_sentiment_pipeline.py
#--------------------------------
"""
Builds an end-to-end Machine Learning pipeline using CountVectorizer 
and Multinomial Naive Bayes for Sentiment Analysis.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd

# 1. Dataset setup
data = {
    "review": [
        "I loved this movie! It was fantastic and full of energy.",
        "Worst movie ever. Total waste of time and money.",
        "The acting was amazing, a truly brilliant masterpiece.",
        "Boring plot, terrible directing, and awful acting.",
        "Highly recommended! An absolute joy to watch.",
        "I hated it. The story made no sense whatsoever.",
    ],
    "Category": [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
    ],
}

df = pd.DataFrame(data)
df["label"] = df["Category"].map({"positive": 1, "negative": 0})

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df["review"],
    df["label"],
    test_size=0.33,
    random_state=42,
    stratify=df["label"],
)

# 3. Create Pipeline
clf = Pipeline(
    [
        ("vectorizer", CountVectorizer()),
        ("nb", MultinomialNB()),
    ]
)

# 4. Train Model
clf.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = clf.predict(X_test)

print("--- MODEL EVALUATION REPORT ---")
print(
    classification_report(
        y_test, y_pred, target_names=["Negative", "Positive"]
    )
)