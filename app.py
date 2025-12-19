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
    # Hugging Face Transformer model (po navodilih)
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1  # CPU (Render free)
    )


# -------------------------
# Sentiment (RAM-safe)
# -------------------------
def run_sentiment_chunked(texts: list[str], chunk_size: int = 40) -> pd.DataFrame:
    """
    Chunked sentiment inference to avoid exceeding 512MB RAM on Render.
    No Streamlit caching of predictions.
    """
    pipe = load_sentiment_model()
    rows = []

    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        preds = pipe(
            chunk,
            batch_size=4,
            truncation=True,
            max_length=128,
        )
        rows.extend(preds)

    return pd.DataFrame({
        "sentiment": [p.get("label", "") for p in rows],
        "confidence": [float(p.get("score", 0.0)) for p in rows],
    })


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


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="HW3 Brand Reputation Monitor (2023)", layout="wide")

st.title("Brand Reputation Monitor (2023)")
st.caption(
    "Scrapes Products/Testimonials/Reviews from web-scraping.dev, "
    "runs Hugging Face Transformer sentiment analysis on 2023 reviews."
)

section = st.sidebar.radio("Navigate", ["Products", "Testimonials", "Reviews"])

# Load data
products_df = load_csv(f"{DATA_DIR}/products.csv")
testimonials_df = load_csv(f"{DATA_DIR}/testimonials.csv")
reviews_df = load_csv(f"{DATA_DIR}/reviews.csv")

# Normalize review dates
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

    show_all = st.checkbox("Show all months (Jan–Dec 2023)", value=False)
    months = month_options_2023() if show_all else available_months_from_data(reviews_df)

    selected = st.select_slider("Select month", options=months, value=months[0])
    start, end = month_to_range(selected)

    filtered = reviews_df[
        (reviews_df["date"] >= start) & (reviews_df["date"] < end)
    ].copy()

    filtered["text"] = filtered["text"].fillna("").astype(str)

    st.write(f"Showing reviews from **{selected}** — rows: **{len(filtered)}**")

    if filtered.empty:
        st.warning("No reviews found for this month.")
        st.stop()

    run_now = st.button("Run sentiment analysis for this month")

    if not run_now:
        st.info("Click the button to run sentiment analysis.")
        st.stop()

    with st.spinner("Running sentiment analysis..."):
        pred_df = run_sentiment_chunked(filtered["text"].tolist())

    filtered = filtered.reset_index(drop=True)
    filtered["sentiment"] = pred_df["sentiment"]
    filtered["confidence"] = pred_df["confidence"]

    # Summary
    summary = (
        filtered.groupby("sentiment")
        .agg(
            count=("sentiment", "size"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total reviews", len(filtered))
    c2.metric("Avg confidence", f"{filtered['confidence'].mean():.3f}")
    c3.metric("Positive share", f"{filtered['sentiment'].eq('POSITIVE').mean()*100:.1f}%")

    # Bar chart (Advanced requirement)
    chart = (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x="sentiment:N",
            y="count:Q",
            tooltip=[
                "sentiment:N",
                "count:Q",
                alt.Tooltip("avg_confidence:Q", format=".3f")
            ],
        )
        .properties(height=300)
    )

    st.altair_chart(chart, width="stretch")

    st.markdown("**Filtered reviews (with predictions)**")
    st.dataframe(
        filtered[["date", "title", "text", "sentiment", "confidence"]]
        .sort_values("date", ascending=False),
        width="stretch",
    )
