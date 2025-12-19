import os
import re
import time
from typing import Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://web-scraping.dev"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) HW3-Scraper/1.0",
    "Accept-Language": "en-US,en;q=0.9",
}

DATA_DIR = "data"


# -----------------------------
# Helpers
# -----------------------------
def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    r = session.get(url, timeout=25, headers=HEADERS)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def parse_total_pages(soup: BeautifulSoup) -> int:
    text = soup.get_text(" ", strip=True)
    m = re.search(r"in\s+(\d+)\s+pages", text, flags=re.IGNORECASE)
    return int(m.group(1)) if m else 1


def safe_json(resp: requests.Response):
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "application/json" in ctype:
        try:
            return resp.json()
        except Exception:
            return None
    return None


# -----------------------------
# Products
# -----------------------------
def scrape_products(session: requests.Session) -> pd.DataFrame:
    print("🔎 Scraping products (HTML pagination)...")

    first = get_soup(f"{BASE}/products", session)
    total_pages = parse_total_pages(first)

    rows: List[Dict] = []
    for p in range(1, total_pages + 1):
        url = f"{BASE}/products?page={p}" if p > 1 else f"{BASE}/products"
        soup = get_soup(url, session)

        # Product detail links contain /product/
        for a in soup.select('a[href*="/product/"]'):
            href = (a.get("href") or "").strip()
            name = a.get_text(strip=True)
            if not href or not name:
                continue

            full_url = BASE + href if href.startswith("/") else href

            # Try to locate a price nearby (optional)
            price = ""
            parent = a
            for _ in range(4):
                if not hasattr(parent, "parent") or parent.parent is None:
                    break
                parent = parent.parent
                txt = parent.get_text(" ", strip=True)
                m = re.search(r"(\d+\.\d{2})", txt)
                if m:
                    price = m.group(1)
                    break

            rows.append(
                {
                    "product_name": name,
                    "product_url": full_url,
                    "price": price,
                }
            )

        time.sleep(0.2)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["product_url"]).reset_index(drop=True)

    print(f"✅ Products scraped: {len(df)}")
    return df


# -----------------------------
# Testimonials
# -----------------------------

def scrape_testimonials(session: requests.Session) -> pd.DataFrame:
    print("🔎 Scraping testimonials (HTMX XHR /api/testimonials paging)...")

    rows: List[Dict] = []
    page = 1

    htmx_headers = {
        **HEADERS,
        "Accept": "*/*",
        "Referer": f"{BASE}/testimonials",
        "Hx-Request": "true",
        "Hx-Current-Url": f"{BASE}/testimonials",
    }

    while True:
        url = f"{BASE}/api/testimonials?page={page}"
        r = session.get(url, timeout=25, headers=htmx_headers)

        if r.status_code != 200:
            print(f"⚠️ Stopping at page {page} (HTTP {r.status_code})")
            # helpful debug preview
            print("Response preview:", r.text[:200].replace("\n", " "))
            break

        soup = BeautifulSoup(r.text, "lxml")

        cards = soup.select("article, .card, .testimonial, [class*='testimonial']")
        # If no cards, we're done
        if not cards:
            print(f"🛑 No more testimonials at page {page}")
            break

        page_rows = 0
        for c in cards:
            text = ""

            bq = c.select_one("blockquote")
            if bq:
                text = bq.get_text(" ", strip=True)

            if not text:
                ps = [p.get_text(" ", strip=True) for p in c.select("p")]
                ps = [t for t in ps if t and len(t) >= 20]
                if ps:
                    text = max(ps, key=len)

            if not text:
                # last resort
                text = c.get_text(" ", strip=True)

            if not text or len(text) < 20:
                continue

            rows.append({"testimonial_text": text})
            page_rows += 1

        print(f"  page {page}: +{page_rows}")
        page += 1
        time.sleep(0.2)

    # ✅ Safe even if empty
    df = pd.DataFrame(rows, columns=["testimonial_text"])
    df["testimonial_text"] = df["testimonial_text"].fillna("").astype(str)
    df = df[df["testimonial_text"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["testimonial_text"]).reset_index(drop=True)

    print(f"✅ Testimonials scraped: {len(df)}")
    return df


# -----------------------------
# Reviews (GraphQL cursor paging)
# -----------------------------

def scrape_reviews(session: requests.Session) -> pd.DataFrame:
    print("🔎 Scraping reviews (GraphQL cursor pagination)...")

    gql_url = f"{BASE}/api/graphql"

    # IMPORTANT: 'title' does NOT exist on Review type (your error proved it)
    query = """
    query GetReviews($first: Int, $after: String) {
      reviews(first: $first, after: $after) {
        edges {
          node {
            date
            text
          }
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
    }
    """

    headers = {
        **HEADERS,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Origin": BASE,
        "Referer": f"{BASE}/reviews",
    }

    all_rows: List[Dict] = []
    after_cursor = None  # start without after

    for batch in range(1, 300):
        variables = {"first": 20}
        # key: OMIT after entirely if we don't have it
        if after_cursor:
            variables["after"] = after_cursor

        payload = {"query": query, "variables": variables}

        resp = session.post(gql_url, json=payload, headers=headers, timeout=25)

        if resp.status_code != 200:
            print(f"⚠️ GraphQL HTTP {resp.status_code} on batch {batch}")
            print("Response preview:", resp.text[:300].replace("\n", " "))
            break

        data = safe_json(resp)
        if not isinstance(data, dict):
            print("⚠️ GraphQL did not return JSON on batch", batch)
            print("Content-Type:", resp.headers.get("Content-Type"))
            print("Response preview:", resp.text[:300].replace("\n", " "))
            break

        if "errors" in data:
            print("⚠️ GraphQL returned errors on batch", batch)
            print(str(data["errors"])[:800])
            break

        reviews_obj = (data.get("data") or {}).get("reviews") or {}
        edges = reviews_obj.get("edges") or []
        page_info = reviews_obj.get("pageInfo") or {}

        if not edges:
            print(f"🛑 No more edges at batch {batch}.")
            break

        batch_rows = []
        for e in edges:
            node = (e or {}).get("node") or {}
            batch_rows.append(
                {
                    "date_raw": node.get("date") or "",
                    "text": node.get("text") or "",
                }
            )

        all_rows.extend(batch_rows)
        print(f"  batch {batch}: +{len(batch_rows)}")

        if not page_info.get("hasNextPage"):
            break

        after_cursor = page_info.get("endCursor") or ""
        if not after_cursor:
            break

        time.sleep(0.2)

    df = pd.DataFrame(all_rows, columns=["date_raw", "text"])
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["date_raw", "text"]).reset_index(drop=True)
    df["title"] = df["text"].str.split().str[:8].str.join(" ") + "..."

    print(f"✅ Reviews scraped: {len(df)} (dates parsed: {df['date'].notna().sum()})")
    return df

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_data_dir()

    with requests.Session() as session:
        session.headers.update(HEADERS)

        products_df = scrape_products(session)
        testimonials_df = scrape_testimonials(session)
        reviews_df = scrape_reviews(session)

    products_path = os.path.join(DATA_DIR, "products.csv")
    testimonials_path = os.path.join(DATA_DIR, "testimonials.csv")
    reviews_path = os.path.join(DATA_DIR, "reviews.csv")

    products_df.to_csv(products_path, index=False)
    testimonials_df.to_csv(testimonials_path, index=False)
    reviews_df.to_csv(reviews_path, index=False)

    print("\n📁 Saved files:")
    print(f" - {products_path}")
    print(f" - {testimonials_path}")
    print(f" - {reviews_path}")


if __name__ == "__main__":
    main()
