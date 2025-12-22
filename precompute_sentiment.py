import pandas as pd
from transformers import pipeline

# Load reviews
df = pd.read_csv("data/reviews.csv")

# Normalize date
if "date" not in df.columns:
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")

df["text"] = df["text"].fillna("").astype(str)

# Load HF transformer (lahko tudi full distilbert tukaj!)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Run sentiment
preds = sentiment_pipe(df["text"].tolist(), batch_size=16, truncation=True)

df["sentiment"] = [p["label"] for p in preds]
df["confidence"] = [float(p["score"]) for p in preds]

# Save enriched file
df.to_csv("data/reviews_with_sentiment.csv", index=False)

print("✅ Sentiment analysis completed and saved.")
