import pandas as pd
import streamlit as st
import altair as alt

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re

DATA_DIR = "data"


# -------------------------
# Caching
# -------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


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


def clean_text_for_wc(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove urls
    text = re.sub(r"[^a-z0-9ščćžđáéíóúàèìòùäëïöüß\s]", " ", text)  # keep letters/numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_wordcloud(text: str, stopwords: set) -> WordCloud:
    return WordCloud(
        width=1200,
        height=500,
        background_color="white",
        stopwords=stopwords,
        collocations=False,
    ).generate(text)


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="HW3 Brand Reputation Monitor (2023)", layout="wide")

st.title("Brand Reputation Monitor (2023)")
st.caption(
    "Scrapes Products/Testimonials/Reviews from web-scraping.dev. "
    "Sentiment predictions were precomputed locally using a Hugging Face Transformer model "
    "and stored in a CSV file for lightweight deployment."
)

section = st.sidebar.radio("Navigate", ["Products", "Testimonials", "Reviews"])

# Load data
products_df = load_csv(f"{DATA_DIR}/products.csv")
testimonials_df = load_csv(f"{DATA_DIR}/testimonials.csv")
reviews_df = load_csv(f"{DATA_DIR}/reviews_with_sentiment.csv")  # enriched locally

# Normalize review dates
if "date" in reviews_df.columns:
    reviews_df["date"] = pd.to_datetime(reviews_df["date"], errors="coerce")
elif "date_raw" in reviews_df.columns:
    reviews_df["date"] = pd.to_datetime(reviews_df["date_raw"], errors="coerce")
else:
    reviews_df["date"] = pd.NaT

# Ensure required columns exist
for col in ["text", "title", "sentiment", "confidence"]:
    if col not in reviews_df.columns:
        reviews_df[col] = "" if col != "confidence" else 0.0

reviews_df["text"] = reviews_df["text"].fillna("").astype(str)
reviews_df["title"] = reviews_df["title"].fillna("").astype(str)
reviews_df["sentiment"] = reviews_df["sentiment"].fillna("").astype(str).str.upper()
reviews_df["confidence"] = pd.to_numeric(reviews_df["confidence"], errors="coerce").fillna(0.0)


# -------------------------
# Sections
# -------------------------
if section == "Products":
    st.subheader("Products")
    st.write(f"Rows: {len(products_df)}")
    st.dataframe(products_df, use_container_width=True)

elif section == "Testimonials":
    st.subheader("Testimonials")
    st.write(f"Rows: {len(testimonials_df)}")
    st.dataframe(testimonials_df, use_container_width=True)

else:
    st.subheader("Reviews — Sentiment Analysis (2023)")

    # Month selector
    show_all = st.checkbox("Show all months (Jan–Dec 2023)", value=False)
    months = month_options_2023() if show_all else available_months_from_data(reviews_df)

    selected = st.select_slider("Select month", options=months, value=months[0])
    start, end = month_to_range(selected)

    # Filter reviews
    filtered = reviews_df[(reviews_df["date"] >= start) & (reviews_df["date"] < end)].copy()
    filtered = filtered.dropna(subset=["date"])

    st.write(f"Showing reviews from **{selected}** — rows: **{len(filtered)}**")

    if filtered.empty:
        st.warning("No reviews found for this month.")
        st.stop()

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Total reviews", len(filtered))
    c2.metric("Avg confidence (overall)", f"{filtered['confidence'].mean():.3f}")
    c3.metric("Positive share", f"{(filtered['sentiment'].eq('POSITIVE').mean() * 100):.1f}%")

    # Summary: count + avg confidence
    summary = (
        filtered.groupby("sentiment")
        .agg(
            count=("sentiment", "size"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )

    # Ensure both labels appear (nice for chart consistency)
    wanted = pd.DataFrame({"sentiment": ["POSITIVE", "NEGATIVE"]})
    summary = wanted.merge(summary, on="sentiment", how="left").fillna({"count": 0, "avg_confidence": 0.0})

    # Bar chart with avg confidence tooltip (Advanced requirement)
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
    st.altair_chart(chart, use_container_width=True)

    # -------------------------
    # BONUS: Word Cloud
    # -------------------------
    st.markdown("### Word Cloud (selected month)")

    wc_text = (
        filtered["title"].fillna("").astype(str) + " " + filtered["text"].fillna("").astype(str)
    ).str.cat(sep=" ")

    wc_text = clean_text_for_wc(wc_text)

    custom_stopwords = {
        "product", "products", "buy", "bought", "use", "used", "using",
        "would", "also", "really", "one", "like", "get", "got", "just",
    }
    stopwords = set(STOPWORDS) | custom_stopwords

    if wc_text.strip():
        wc = make_wordcloud(wc_text, stopwords)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Not enough text to generate a word cloud for this month.")

    # Table
    st.markdown("**Filtered reviews (with predictions)**")
    st.dataframe(
        filtered[["date", "title", "text", "sentiment", "confidence"]].sort_values("date", ascending=False),
        use_container_width=True,
    )
