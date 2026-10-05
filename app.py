import streamlit as st
import joblib
import re
import pandas as pd
import plotly.graph_objects as go

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="CineSense 5★",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model     = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def preprocess_text(text):
    text  = text.lower()
    text  = re.sub(r"<.*?>",      " ", text)
    text  = re.sub(r"[^a-zA-Z\s]"," ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


def analyze(review_text):
    vec        = vectorizer.transform([preprocess_text(review_text)])
    rating     = int(model.predict(vec)[0])
    probs      = model.predict_proba(vec)[0]
    confidence = round(float(max(probs)) * 100, 2)
    sentiment  = {5:"😍 Highly Positive", 4:"😊 Positive",
                  3:"😐 Neutral",         2:"😕 Negative",
                  1:"😞 Highly Negative"}[rating]
    return rating, confidence, sentiment, probs


PASTEL = ["#F4A7B9","#F9C784","#A8D8EA","#B5EAD7","#C9A7EB"]


# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown("""
<style>

/* ---- DARK BACKGROUND ---- */
.stApp { background-color: #0F0F0F; }
[data-testid="stSidebar"] { background-color: #1A1A2E; }

/* ---- FORCE ALL TEXT WHITE ---- */
* { color: #E8E8E8 !important; }

.main-title {
    text-align: center; color: #C9A7EB !important;
    font-size: 40px; font-weight: 700; margin-bottom: 4px;
}
.subtitle {
    text-align: center; color: #9B8AAA !important;
    font-size: 15px; margin-bottom: 28px;
}

.card {
    background: #1E1E2E; border-radius: 20px;
    padding: 26px; margin-bottom: 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.4);
}

.result-card {
    background: linear-gradient(135deg,#2A1A3E,#1E1030);
    border-radius: 22px; padding: 30px;
    text-align: center; margin-top: 18px;
    box-shadow: 0 4px 18px rgba(180,140,210,0.2);
}
.star-display  { font-size: 36px; letter-spacing: 6px; margin: 8px 0; }
.rating-big    { font-size: 30px; font-weight: 700; color: #C9A7EB !important; }
.sent-label    { font-size: 19px; color: #B48FD8 !important; margin: 6px 0; }
.conf-text     { color: #9B8AAA !important; font-size: 14px; }

.stat-row { display:flex; gap:14px; margin-top:18px; flex-wrap:wrap; }
.stat-box {
    flex:1; min-width:100px; background:#1E1E2E;
    border-radius:16px; padding:16px 10px;
    text-align:center; box-shadow:0 3px 12px rgba(0,0,0,0.3);
}
.stat-icon { font-size:20px; }
.stat-val  { font-size:22px; font-weight:700; color:#C9A7EB !important; margin:4px 0; }
.stat-lbl  { font-size:11px; color:#9B8AAA !important; }

.dash-card {
    background:#1E1E2E; border-radius:18px; padding:20px;
    text-align:center; box-shadow:0 3px 14px rgba(0,0,0,0.3);
}
.dash-icon { font-size:26px; }
.dash-val  { font-size:28px; font-weight:700; color:#C9A7EB !important; margin:5px 0 2px; }
.dash-lbl  { font-size:12px; color:#9B8AAA !important; }

.hist-row {
    background:#1E1E2E; border-radius:14px; padding:13px 18px;
    margin-bottom:9px; display:flex;
    justify-content:space-between; align-items:center;
    box-shadow:0 2px 10px rgba(0,0,0,0.3);
}
.hist-num    { color:#555577 !important; font-size:12px; margin-right:12px; min-width:24px; }
.hist-text   { color:#E8E8E8 !important; font-size:13px; flex:1; }
.hist-badge  {
    background:#2A1A3E; color:#C9A7EB !important;
    border-radius:20px; padding:3px 12px;
    font-size:12px; font-weight:600;
    margin-left:10px; white-space:nowrap;
}
.hist-conf   {
    background:#0F2A1E; color:#4ABA7A !important;
    border-radius:20px; padding:3px 12px;
    font-size:12px; font-weight:600;
    margin-left:6px; white-space:nowrap;
}

div.stButton > button {
    background: linear-gradient(135deg,#7B2FBE,#5A189A);
    color:#FFFFFF !important; border:none; border-radius:14px;
    padding:11px 26px; font-size:15px; font-weight:600; width:100%;
}
div.stButton > button:hover {
    background: linear-gradient(135deg,#9B4FDE,#7B2FBE);
    color:#FFFFFF !important;
}

/* Inputs */
.stTextArea textarea {
    background-color: #1E1E2E !important;
    color: #E8E8E8 !important;
    border: 1px solid #3A3A5A !important;
    border-radius: 12px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #1E1E2E !important;
    border-radius: 12px !important;
    border: 1px dashed #3A3A5A !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { background-color: #1E1E2E !important; }

.divider { border:none; border-top:1.5px solid #2A2A4A; margin:22px 0; }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div style="text-align:center;font-size:30px;margin-bottom:2px;">🎬</div>'
        '<div style="text-align:center;font-weight:700;color:#C9A7EB;font-size:17px;">CineSense 5★</div>'
        '<div style="text-align:center;color:#9B8AAA;font-size:11px;margin-bottom:24px;">NLP Sentiment Analysis</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "nav", ["🔍 Analyze", "📦 Batch", "📊 Dashboard", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#DDD0EA;margin:20px 0;'>", unsafe_allow_html=True)

    total      = len(st.session_state.history)
    avg_rating = round(sum(r["rating"] for r in st.session_state.history)/total,1) if total else "—"

    st.markdown(
        f'<div style="color:#9B8AAA;font-size:11px;text-align:center;">Reviews analyzed</div>'
        f'<div style="color:#C9A7EB;font-size:24px;font-weight:700;text-align:center;">{total}</div>'
        f'<div style="color:#9B8AAA;font-size:11px;text-align:center;margin-top:10px;">Avg rating</div>'
        f'<div style="color:#C9A7EB;font-size:24px;font-weight:700;text-align:center;">{avg_rating} ★</div>',
        unsafe_allow_html=True
    )


# ==================================================
# PAGE — ANALYZE
# ==================================================

if page == "🔍 Analyze":

    st.markdown('<div class="main-title">🎬 CineSense 5★</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multi-Class Sentiment Analysis for Movie Reviews</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="card">✨ Enter a movie review and CineSense will predict its rating '
        'from <b>1★ to 5★</b>.</div>',
        unsafe_allow_html=True
    )

    review = st.text_area(
        "🎥 Your Movie Review",
        placeholder="Example: The story was beautiful and the acting was excellent...",
        height=155
    )

    if st.button("✨ Analyze Review"):

        if not review.strip():
            st.warning("Please enter a movie review first.")

        else:
            rating, confidence, sentiment, probs = analyze(review)
            stars = "⭐" * rating

            st.session_state.history.append({
                "review":     review[:80] + ("..." if len(review) > 80 else ""),
                "rating":     rating,
                "sentiment":  sentiment,
                "confidence": confidence,
                "words":      len(review.split()),
                "chars":      len(review)
            })

            # Result card
            st.markdown(
                f'<div class="result-card">'
                f'<div style="font-size:12px;color:#9B8AAA;letter-spacing:2px;text-transform:uppercase;">Predicted Rating</div>'
                f'<div class="star-display">{stars}</div>'
                f'<div class="rating-big">{rating} / 5</div>'
                f'<div class="sent-label">{sentiment}</div>'
                f'<div class="conf-text">Confidence: <b>{confidence:.2f}%</b></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Stat boxes
            st.markdown(
                f'<div class="stat-row">'
                f'<div class="stat-box"><div class="stat-icon">⭐</div><div class="stat-val">{rating}/5</div><div class="stat-lbl">Rating</div></div>'
                f'<div class="stat-box"><div class="stat-icon">🎯</div><div class="stat-val">{confidence:.1f}%</div><div class="stat-lbl">Confidence</div></div>'
                f'<div class="stat-box"><div class="stat-icon">📝</div><div class="stat-val">{len(review.split())}</div><div class="stat-lbl">Words</div></div>'
                f'<div class="stat-box"><div class="stat-icon">🔤</div><div class="stat-val">{len(review)}</div><div class="stat-lbl">Characters</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Probability chart
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("**📊 Rating Probability Distribution**")

            fig = go.Figure(go.Bar(
                x=[f"{int(c)}★" for c in model.classes_],
                y=[round(p * 100, 2) for p in probs],
                marker_color=PASTEL,
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="outside"
            ))
            fig.update_layout(
                plot_bgcolor="#0F0F0F", paper_bgcolor="#0F0F0F",
                yaxis=dict(title="Probability (%)", range=[0, 115], color="#E8E8E8"),
                xaxis=dict(title="Rating", color="#E8E8E8"),
                font=dict(color="#E8E8E8", size=13),
                margin=dict(t=10, b=10, l=10, r=10),
                height=310
            )
            st.plotly_chart(fig, use_container_width=True)


# ==================================================
# PAGE — BATCH
# ==================================================

elif page == "📦 Batch":

    st.markdown('<div class="main-title">📦 Batch Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyze multiple reviews at once</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="card">Upload a <b>CSV file</b> with a column named <b>review</b>. '
        'CineSense will predict the rating for every row.</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)

        if "review" not in df_up.columns:
            st.error("CSV must have a column named 'review'.")

        else:
            df_up = df_up.dropna(subset=["review"])

            results = []
            for text in df_up["review"]:
                r, c, s, _ = analyze(str(text))
                results.append({"review": str(text)[:70], "rating": r,
                                 "sentiment": s, "confidence": f"{c}%"})

            df_result = pd.DataFrame(results)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown(f"**✅ {len(df_result)} reviews analyzed**")
            st.dataframe(df_result, use_container_width=True, hide_index=True)

            # Add to history
            for row in results:
                st.session_state.history.append({
                    "review":     row["review"],
                    "rating":     row["rating"],
                    "sentiment":  row["sentiment"],
                    "confidence": float(row["confidence"].replace("%","")),
                    "words":      len(row["review"].split()),
                    "chars":      len(row["review"])
                })

            # Export
            csv_out = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results as CSV",
                data=csv_out,
                file_name="cinesense_results.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        st.markdown(
            '<div class="card" style="text-align:center;color:#9B8AAA;padding:40px;">'
            '📂 Upload a CSV file to get started</div>',
            unsafe_allow_html=True
        )


# ==================================================
# PAGE — DASHBOARD
# ==================================================

elif page == "📊 Dashboard":

    st.markdown('<div class="main-title">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Overview of all analyzed reviews</div>', unsafe_allow_html=True)

    history = st.session_state.history

    if not history:
        st.markdown(
            '<div class="card" style="text-align:center;color:#9B8AAA;padding:50px;">'
            '🎬 No reviews yet. Go to <b>Analyze</b> or <b>Batch</b> to get started!</div>',
            unsafe_allow_html=True
        )

    else:
        total      = len(history)
        avg_rating = round(sum(r["rating"] for r in history) / total, 1)
        avg_conf   = round(sum(r["confidence"] for r in history) / total, 1)
        top_sent   = max(
            set(r["sentiment"] for r in history),
            key=lambda s: sum(1 for r in history if r["sentiment"] == s)
        ).split(" ", 1)[-1]

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        for col, icon, val, lbl in [
            (c1, "🎬", total,           "Reviews Analyzed"),
            (c2, "⭐", f"{avg_rating}/5","Avg Rating"),
            (c3, "🎯", f"{avg_conf}%",  "Avg Confidence"),
            (c4, "💬", top_sent,        "Top Sentiment"),
        ]:
            with col:
                st.markdown(
                    f'<div class="dash-card"><div class="dash-icon">{icon}</div>'
                    f'<div class="dash-val">{val}</div><div class="dash-lbl">{lbl}</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Charts
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**⭐ Rating Distribution**")
            rc = {i: sum(1 for r in history if r["rating"] == i) for i in range(1, 6)}
            fig1 = go.Figure(go.Bar(
                x=[f"{k}★" for k in rc],
                y=list(rc.values()),
                marker_color=PASTEL,
                text=list(rc.values()),
                textposition="outside"
            ))
            fig1.update_layout(
                plot_bgcolor="#0F0F0F", paper_bgcolor="#0F0F0F",
                yaxis=dict(title="Count", tickformat="d",
                           range=[0, max(rc.values()) + 1], color="#E8E8E8"),
                xaxis=dict(color="#E8E8E8"),
                font=dict(color="#E8E8E8", size=13),
                margin=dict(t=10, b=10, l=10, r=10), height=290
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_r:
            st.markdown("**🥧 Sentiment Breakdown**")
            sc = {}
            for r in history:
                sc[r["sentiment"]] = sc.get(r["sentiment"], 0) + 1
            fig2 = go.Figure(go.Pie(
                labels=list(sc.keys()),
                values=list(sc.values()),
                marker_colors=PASTEL,
                hole=0.4,
                textinfo="label+percent"
            ))
            fig2.update_layout(
                paper_bgcolor="#0F0F0F",
                font=dict(color="#E8E8E8", size=12),
                margin=dict(t=10, b=10, l=10, r=10),
                height=290, showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Review history list
        st.markdown("**📋 Review History**")
        for i, r in enumerate(reversed(history), 1):
            idx = len(history) - i + 1
            st.markdown(
                f'<div class="hist-row">'
                f'<span class="hist-num">#{idx}</span>'
                f'<span class="hist-text">{r["review"]}</span>'
                f'<span class="hist-badge">{"⭐"*r["rating"]} {r["rating"]}/5</span>'
                f'<span class="hist-conf">{r["confidence"]}%</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Export history
        st.markdown("<br>", unsafe_allow_html=True)
        col_exp, col_clr = st.columns(2)

        with col_exp:
            df_hist = pd.DataFrame(history)[["review","rating","sentiment","confidence","words","chars"]]
            st.download_button(
                "⬇️ Export History as CSV",
                data=df_hist.to_csv(index=False).encode("utf-8"),
                file_name="cinesense_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_clr:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()


# ==================================================
# PAGE — ABOUT
# ==================================================

elif page == "ℹ️ About":

    st.markdown('<div class="main-title">ℹ️ About CineSense 5★</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">How it works</div>', unsafe_allow_html=True)

    for content in [
        ("<b>🎬 What is CineSense?</b><br><br>"
         "CineSense is an NLP-powered movie review sentiment analyzer. "
         "It reads a review and predicts a star rating from <b>1★ to 5★</b> using machine learning."),

        ("<b>⚙️ How it works</b><br><br>"
         "1. Your review is <b>preprocessed</b> — lowercased, cleaned, stopwords removed, lemmatized<br>"
         "2. <b>TF-IDF</b> converts the text into numerical features<br>"
         "3. A <b>Logistic Regression</b> model predicts the rating<br>"
         "4. Confidence scores are shown for all 5 rating classes"),

        ("<b>📁 Pages</b><br><br>"
         "• <b>🔍 Analyze</b> — Single review analysis with probability chart<br>"
         "• <b>📦 Batch</b> — Upload CSV, analyze many reviews, download results<br>"
         "• <b>📊 Dashboard</b> — Rating distribution, sentiment breakdown, history<br>"
         "• <b>ℹ️ About</b> — Project info"),
    ]:
        st.markdown(f'<div class="card">{content}</div>', unsafe_allow_html=True)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    '<div style="text-align:center;color:#C0B0CC;font-size:12px;margin-top:36px;">'
    '🎬 CineSense 5★ — NLP Sentiment Analysis Project'
    '</div>',
    unsafe_allow_html=True
)
