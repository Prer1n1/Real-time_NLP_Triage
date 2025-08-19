import time
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
try:
    API = st.secrets["API_URL"]
except Exception:
    pass

st.set_page_config(page_title="Realtime NLP Triage", layout="wide")
st.title(" Realtime NLP Triage Dashboard")

with st.sidebar:
    st.header("Filters")
    q = st.text_input("Keyword contains", "")
    sentiment = st.selectbox("Sentiment", ["", "positive", "negative", "neutral"])
    intent = st.selectbox(
        "Intent",
        ["", "refund_request", "order_status", "cancel_order", "product_issue", "billing_issue", "complaint", "praise", "other"],
    )
    toxic = st.selectbox("Toxic", ["", "true", "false"])
    language = st.text_input("Language code (e.g., en, es)", "")
    limit = st.slider("Rows to fetch", 10, 200, 100, 10)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_secs = st.slider("Refresh every (seconds)", 2, 30, 5)
    st.caption("API: " + API)

def fetch_metrics():
    try:
        r = requests.get(f"{API}/metrics", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to fetch /metrics: {e}")
        return {}

def fetch_rows():
    params = {"limit": limit}
    if q: params["q"] = q
    if sentiment: params["sentiment"] = sentiment
    if intent: params["intent"] = intent
    if toxic: params["toxic"] = (toxic == "true")
    if language: params["language"] = language
    try:
        r = requests.get(f"{API}/search", params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception as e:
        st.error(f"Failed to fetch /search: {e}")
        return []

# ---- Layout: top KPs
col1, col2, col3, col4 = st.columns(4, gap="large")
metrics = fetch_metrics()
with col1:
    st.metric("Total messages", metrics.get("total_messages", "—"))
with col2:
    st.metric("Toxic", metrics.get("toxic_count", "—"))
with col3:
    st.metric("Negative", metrics.get("sentiment_counts", {}).get("negative", "—"))
with col4:
    st.metric("Refund intents", metrics.get("intent_counts", {}).get("refund_request", "—"))

# ---- Charts
rows = fetch_rows()
df = pd.DataFrame(rows)

if df.empty:
    st.info("No data yet. Send a few messages via /client or POST /analyze.")
else:
    # Left: sentiment distribution, Right: intent distribution
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("Sentiment distribution")
        s_counts = df["sentiment_label"].value_counts(dropna=False).sort_index()
        fig1, ax1 = plt.subplots()
        s_counts.plot(kind="bar", ax=ax1)
        ax1.set_xlabel("Sentiment")
        ax1.set_ylabel("Count")
        ax1.set_title("Messages by sentiment")
        st.pyplot(fig1, clear_figure=True)

    with c2:
        st.subheader("Intent distribution")
        if "intent_label" in df.columns:
            i_counts = df["intent_label"].value_counts(dropna=False).sort_values(ascending=False)
            fig2, ax2 = plt.subplots()
            i_counts.plot(kind="bar", ax=ax2)
            ax2.set_xlabel("Intent")
            ax2.set_ylabel("Count")
            ax2.set_title("Messages by intent")
            st.pyplot(fig2, clear_figure=True)
        else:
            st.write("No intent data found.")

    # Toxic over time
    st.subheader("Toxic vs non-toxic over time (latest rows)")
    if "created_at" in df.columns:
        dft = df.copy()
        dft["created_at"] = pd.to_datetime(dft["created_at"], errors="coerce")
        dft = dft.sort_values("created_at")
        dft["toxic_flag"] = dft["toxicity_is_toxic"].fillna(False).astype(int)
        dft["non_toxic_flag"] = 1 - dft["toxic_flag"]
        # rolling sum window (just for viz smoothing)
        window = max(1, min(20, len(dft)//5))
        dft["toxic_rolling"] = dft["toxic_flag"].rolling(window=window).sum()
        dft["nontoxic_rolling"] = dft["non_toxic_flag"].rolling(window=window).sum()

        fig3, ax3 = plt.subplots()
        ax3.plot(dft["created_at"], dft["toxic_rolling"], label="toxic (rolling)")
        ax3.plot(dft["created_at"], dft["nontoxic_rolling"], label="non-toxic (rolling)")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Count (rolling)")
        ax3.set_title("Toxicity trend")
        ax3.legend()
        st.pyplot(fig3, clear_figure=True)
    else:
        st.write("No timestamps available.")

    # Table
    st.subheader("Latest messages")
    show_cols = ["created_at","id","language","sentiment_label","sentiment_score","intent_label","toxicity_is_toxic","text"]
    for c in show_cols:
        if c not in df.columns:
            df[c] = None
    st.dataframe(df[show_cols], use_container_width=True, height=360)

# ---- Auto refresh
if auto_refresh:
    st.caption(f"Auto-refreshing every {refresh_secs} seconds…")
    time.sleep(refresh_secs)
    st.rerun()