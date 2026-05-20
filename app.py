import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
from preprocess import clean_text
from sentiment import get_sentiment

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #f8f9fa; }
    .main { background-color: #ffffff; }

    .title-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    .title-box h1 { font-size: 2.5em; margin: 0; }
    .title-box p  { font-size: 1.1em; opacity: 0.9; margin: 5px 0 0; }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 15px;
    }
    .positive { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .negative { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    .neutral  { background-color: #fff3cd; color: #856404; border: 2px solid #ffeeba; }

    .history-item {
        background: #f1f3f5;
        border-left: 4px solid #667eea;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 0.95em;
    }

    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #dee2e6;
    }

    div[data-testid="stTabs"] button {
        font-size: 1em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State for History ──────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ─── Header ────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>💬 Sentiment Analysis Dashboard</h1>
    <p>Analyze the emotion behind any text — instantly</p>
</div>
""", unsafe_allow_html=True)

# ─── Emoji Helper ──────────────────────────────────────────
def sentiment_emoji(label):
    return {"Positive": "😊 Positive", "Negative": "😞 Negative", "Neutral": "😐 Neutral"}.get(label, label)

def sentiment_class(label):
    return {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"}.get(label, "neutral")

# ─── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["✍️ Single Review", "📂 CSV Bulk Analysis", "🕓 History"])

# ══════════════════════════════════════════════════════════
# TAB 1 — Single Review
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Analyze a Single Review")
    user_input = st.text_area("Enter your text below:", height=150,
                               placeholder="e.g. This product is absolutely amazing!")

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)

    if analyze_btn and user_input.strip():
        # Progress bar
        bar = st.progress(0, text="Cleaning text...")
        time.sleep(0.4)
        bar.progress(40, text="Running sentiment model...")
        cleaned = clean_text(user_input)
        result = get_sentiment(cleaned)
        time.sleep(0.4)
        bar.progress(100, text="Done!")
        time.sleep(0.3)
        bar.empty()

        # Result box
        css_class = sentiment_class(result)
        label = sentiment_emoji(result)
        st.markdown(f'<div class="result-box {css_class}">{label}</div>',
                    unsafe_allow_html=True)

        # Save to history
        st.session_state.history.append({
            "Text": user_input[:60] + ("..." if len(user_input) > 60 else ""),
            "Sentiment": result
        })

    elif analyze_btn:
        st.warning("⚠️ Please enter some text first!")

# ══════════════════════════════════════════════════════════
# TAB 2 — CSV Bulk Analysis
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Bulk Sentiment Analysis via CSV")
    uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("**Preview:**")
        st.dataframe(df.head(), use_container_width=True)

        column = st.selectbox("Select the text column to analyze:", df.columns)

        if st.button("🚀 Run Bulk Analysis", use_container_width=False):
            # Progress bar
            bar = st.progress(0, text="Starting analysis...")
            total = len(df)
            results = []
            for i, text in enumerate(df[column]):
                cleaned = clean_text(text)
                results.append(get_sentiment(cleaned))
                bar.progress(int((i + 1) / total * 100),
                             text=f"Analyzing row {i+1} of {total}...")
            bar.empty()

            df['Sentiment'] = results

            # ── Metrics Row ──
            pos = (df['Sentiment'] == 'Positive').sum()
            neg = (df['Sentiment'] == 'Negative').sum()
            neu = (df['Sentiment'] == 'Neutral').sum()

            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card">😊<br><b style="font-size:1.8em">{pos}</b><br>Positive</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card">😞<br><b style="font-size:1.8em">{neg}</b><br>Negative</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card">😐<br><b style="font-size:1.8em">{neu}</b><br>Neutral</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Charts ──
            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(df, names='Sentiment',
                             color='Sentiment',
                             color_discrete_map={
                                 'Positive': '#28a745',
                                 'Negative': '#dc3545',
                                 'Neutral':  '#ffc107'
                             },
                             title='Sentiment Distribution')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                counts = df['Sentiment'].value_counts().reset_index()
                counts.columns = ['Sentiment', 'Count']
                fig2 = px.bar(counts, x='Sentiment', y='Count',
                              color='Sentiment',
                              color_discrete_map={
                                  'Positive': '#28a745',
                                  'Negative': '#dc3545',
                                  'Neutral':  '#ffc107'
                              },
                              title='Sentiment Count')
                st.plotly_chart(fig2, use_container_width=True)

            # ── Word Cloud ──
            st.subheader("☁️ Word Cloud")
            all_text = " ".join(df[column].astype(str).tolist())
            wc = WordCloud(width=800, height=300,
                           background_color='white',
                           colormap='cool').generate(all_text)
            fig3, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig3)

            # ── Results Table ──
            st.subheader("📋 Detailed Results")
            st.dataframe(df[[column, 'Sentiment']], use_container_width=True)

            # ── Download ──
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results as CSV", csv,
                               "results.csv", "text/csv",
                               use_container_width=False)

# ══════════════════════════════════════════════════════════
# TAB 3 — History
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("🕓 Analysis History (This Session)")

    if st.session_state.history:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

        for item in reversed(st.session_state.history):
            emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}.get(item['Sentiment'], "")
            st.markdown(
                f'<div class="history-item">{emoji} <b>{item["Sentiment"]}</b> — {item["Text"]}</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No analysis done yet. Go to the Single Review tab and analyze some text!")