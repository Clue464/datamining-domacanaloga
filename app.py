import pandas as pd
import streamlit as st
import altair as alt
from transformers import pipeline

DATA_DIR = "data"


# -------------------------
# Caching
# -------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_sentiment_model():
    # rubric-friendly model
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


# -------------------------
# Helpers
# -------------------------
def month_options_2023():
    months = pd.date_range("2023-01-01", "2023-12-01", freq="MS")
    return [m.strftime("%b %Y") for m in months]


def available_months_from_data(reviews_df: pd.DataFrame):
    df = reviews_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df[(df["date"] >= "2023-01-01") & (df["date"] < "2024-01-01")]
    if df.empty:
        return month_options_2023()

    month_starts = df["date"].dt.to_period("M").dt.to_timestamp()
    month_starts = sorted(month_starts.unique())
    return [m.strftime("%b %Y") for m in month_starts]


def month_to_range(label: str):
    start = pd.to_datetime(label)
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def run_sentiment(pipe, texts, batch_size=16):
    return pipe(texts, batch_size=batch_size, truncation=True)


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="HW3 Brand Reputation Monitor (2023)", layout="wide")

st.title("Brand Reputation Monitor (2023)")
st.caption(
    "Scrapes Products/Testimonials/Reviews from web-scraping.dev, runs Transformer sentiment analysis on 2023 reviews."
)

section = st.sidebar.radio("Navigate", ["Products", "Testimonials", "Reviews"])

# Load data
products_df = load_csv(f"{DATA_DIR}/products.csv")
testimonials_df = load_csv(f"{DATA_DIR}/testimonials.csv")
reviews_df = load_csv(f"{DATA_DIR}/reviews.csv")

# Normalize reviews date column
if "date" in reviews_df.columns:
    reviews_df["date"] = pd.to_datetime(reviews_df["date"], errors="coerce")
elif "date_raw" in reviews_df.columns:
    reviews_df["date"] = pd.to_datetime(reviews_df["date_raw"], errors="coerce")
else:
    reviews_df["date"] = pd.NaT

# Ensure text/title columns exist
if "text" not in reviews_df.columns:
    reviews_df["text"] = ""
if "title" not in reviews_df.columns:
    reviews_df["title"] = (
        reviews_df["text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str[:8]
        .str.join(" ")
        + "..."
    )

# -------------------------
# Sections
# -------------------------
if section == "Products":
    st.subheader("Products")
    st.write(f"Rows: {len(products_df)}")
    st.dataframe(products_df, width="stretch")

elif section == "Testimonials":
    st.subheader("Testimonials")
    st.write(f"Rows: {len(testimonials_df)}")
    st.dataframe(testimonials_df, width="stretch")

else:
    st.subheader("Reviews — Sentiment Analysis (2023)")

    # Month selector (rubric-safe toggle)
    show_all = st.checkbox("Show all months (Jan–Dec 2023)", value=False)
    months = month_options_2023() if show_all else available_months_from_data(reviews_df)

    selected = st.select_slider("Select month", options=months, value=months[0])
    start, end = month_to_range(selected)

    # Filter reviews for selected month
    filtered = reviews_df[(reviews_df["date"] >= start) & (reviews_df["date"] < end)].copy()
    filtered = filtered.dropna(subset=["date"])
    filtered["text"] = filtered["text"].fillna("").astype(str)

    st.write(f"Showing reviews from **{selected}** — rows: **{len(filtered)}**")

    if len(filtered) == 0:
        st.warning("No reviews found for this month in your dataset.")
        st.stop()

    # Load model
    with st.spinner("Loading sentiment model..."):
        sent_pipe = load_sentiment_model()

    # Run sentiment
    with st.spinner("Running sentiment analysis on filtered reviews..."):
        preds = run_sentiment(sent_pipe, filtered["text"].tolist(), batch_size=16)

    filtered["sentiment"] = [p.get("label", "") for p in preds]
    filtered["confidence"] = [float(p.get("score", 0.0)) for p in preds]

    # Summary stats (count + avg confidence per sentiment)
    summary = (
        filtered.groupby("sentiment")
        .agg(
            count=("sentiment", "size"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )

    avg_conf_overall = filtered["confidence"].mean()
    avg_conf_by_class = summary[["sentiment", "avg_confidence"]].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total reviews", f"{len(filtered)}")
    c2.metric("Avg confidence (overall)", f"{avg_conf_overall:.3f}")
    c3.metric("Positive share", f"{(filtered['sentiment'].eq('POSITIVE').mean() * 100):.1f}%")

    # Bar chart with tooltip that includes avg confidence (Advanced requirement)
    chart = (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X("sentiment:N", title="Sentiment"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=[
                alt.Tooltip("sentiment:N", title="Sentiment"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("avg_confidence:Q", title="Avg confidence", format=".3f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, width="stretch")

    st.markdown("**Average confidence by class**")
    st.dataframe(avg_conf_by_class, width="stretch")

    st.markdown("**Filtered reviews (with predictions)**")
    show_cols = [c for c in ["date", "title", "text", "sentiment", "confidence"] if c in filtered.columns]
    st.dataframe(filtered[show_cols].sort_values("date", ascending=False), width="stretch")
