import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
from preprocess import clean_text
from sentiment import get_sentiment
import text2emotion as te

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
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

    .emotion-card {
        border-radius: 12px;
        padding: 15px 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1em;
        margin: 5px;
    }
    .happy    { background-color: #fff9c4; color: #f57f17; border: 2px solid #f9a825; }
    .angry    { background-color: #ffcdd2; color: #b71c1c; border: 2px solid #ef9a9a; }
    .surprise { background-color: #e1bee7; color: #6a1b9a; border: 2px solid #ce93d8; }
    .sad      { background-color: #bbdefb; color: #0d47a1; border: 2px solid #90caf9; }
    .fear     { background-color: #cfd8dc; color: #263238; border: 2px solid #90a4ae; }

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
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ─── Header ────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>💬 Sentiment Analysis Dashboard</h1>
    <p>Analyze the emotion behind any text — instantly</p>
</div>
""", unsafe_allow_html=True)

# ─── Helpers ───────────────────────────────────────────────
def sentiment_emoji(label):
    return {"Positive": "😊 Positive",
            "Negative": "😞 Negative",
            "Neutral":  "😐 Neutral"}.get(label, label)

def sentiment_class(label):
    return {"Positive": "positive",
            "Negative": "negative",
            "Neutral":  "neutral"}.get(label, "neutral")

EMOTION_META = {
    "Happy":    ("😊", "happy"),
    "Angry":    ("😤", "angry"),
    "Surprise": ("😲", "surprise"),
    "Sad":      ("😢", "sad"),
    "Fear":     ("😰", "fear"),
}

def show_emotion_cards(text):
    """Detect emotions and show colour coded cards."""
    scores = te.get_emotion(text)          # returns dict
    # filter out zero scores
    active = {k: v for k, v in scores.items() if v > 0}

    if not active:
        st.info("No strong emotion detected in this text.")
        return

    st.markdown("#### 🎭 Emotion Breakdown")
    cols = st.columns(len(active))
    for col, (emotion, score) in zip(cols, active.items()):
        emoji, css = EMOTION_META.get(emotion, ("🔵", "neutral"))
        pct = round(score * 100, 1)
        col.markdown(
            f'<div class="emotion-card {css}">'
            f'{emoji}<br>{emotion}<br>'
            f'<span style="font-size:1.3em">{pct}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )

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
        bar = st.progress(0, text="Cleaning text...")
        time.sleep(0.4)
        bar.progress(40, text="Running sentiment model...")
        cleaned = clean_text(user_input)
        result  = get_sentiment(cleaned)
        time.sleep(0.4)
        bar.progress(80, text="Detecting emotions...")
        time.sleep(0.4)
        bar.progress(100, text="Done!")
        time.sleep(0.3)
        bar.empty()

        # Sentiment result box
        css_class = sentiment_class(result)
        label     = sentiment_emoji(result)
        st.markdown(f'<div class="result-box {css_class}">{label}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Emotion cards
        show_emotion_cards(user_input)

        # Save to history
        st.session_state.history.append({
            "Text":      user_input[:60] + ("..." if len(user_input) > 60 else ""),
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
            bar   = st.progress(0, text="Starting analysis...")
            total = len(df)
            sentiments, happy_l, angry_l, sad_l, surprise_l, fear_l = [], [], [], [], [], []

            for i, text in enumerate(df[column]):
                cleaned = clean_text(str(text))
                sentiments.append(get_sentiment(cleaned))
                emotions = te.get_emotion(str(text))
                happy_l.append(round(emotions.get("Happy",    0) * 100, 1))
                angry_l.append(round(emotions.get("Angry",    0) * 100, 1))
                sad_l.append(  round(emotions.get("Sad",      0) * 100, 1))
                surprise_l.append(round(emotions.get("Surprise", 0) * 100, 1))
                fear_l.append( round(emotions.get("Fear",     0) * 100, 1))
                bar.progress(int((i + 1) / total * 100),
                             text=f"Analyzing row {i+1} of {total}...")
            bar.empty()

            df['Sentiment'] = sentiments
            df['Happy %']   = happy_l
            df['Angry %']   = angry_l
            df['Sad %']     = sad_l
            df['Surprise %']= surprise_l
            df['Fear %']    = fear_l

            # ── Sentiment Metric Cards ──
            pos = (df['Sentiment'] == 'Positive').sum()
            neg = (df['Sentiment'] == 'Negative').sum()
            neu = (df['Sentiment'] == 'Neutral').sum()

            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card">😊<br><b style="font-size:1.8em">{pos}</b><br>Positive</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card">😞<br><b style="font-size:1.8em">{neg}</b><br>Negative</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card">😐<br><b style="font-size:1.8em">{neu}</b><br>Neutral</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Charts Row 1 — Sentiment ──
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(df, names='Sentiment',
                             color='Sentiment',
                             color_discrete_map={
                                 'Positive': '#28a745',
                                 'Negative': '#dc3545',
                                 'Neutral':  '#ffc107'},
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
                                  'Neutral':  '#ffc107'},
                              title='Sentiment Count')
                st.plotly_chart(fig2, use_container_width=True)

            # ── Charts Row 2 — Emotions ──
            st.markdown("#### 🎭 Emotion Distribution Across All Reviews")
            emotion_avg = {
                'Happy':    df['Happy %'].mean(),
                'Angry':    df['Angry %'].mean(),
                'Sad':      df['Sad %'].mean(),
                'Surprise': df['Surprise %'].mean(),
                'Fear':     df['Fear %'].mean(),
            }
            emo_df = pd.DataFrame(list(emotion_avg.items()),
                                  columns=['Emotion', 'Average %'])
            fig3 = px.bar(emo_df, x='Emotion', y='Average %',
                          color='Emotion',
                          color_discrete_map={
                              'Happy':    '#f9a825',
                              'Angry':    '#ef9a9a',
                              'Sad':      '#90caf9',
                              'Surprise': '#ce93d8',
                              'Fear':     '#90a4ae'},
                          title='Average Emotion Scores Across All Reviews')
            st.plotly_chart(fig3, use_container_width=True)

            # ── Word Cloud ──
            st.subheader("☁️ Word Cloud")
            all_text = " ".join(df[column].astype(str).tolist())
            wc  = WordCloud(width=800, height=300,
                            background_color='white',
                            colormap='cool').generate(all_text)
            fig4, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig4)

            # ── Results Table ──
            st.subheader("📋 Detailed Results")
            st.dataframe(df[[column, 'Sentiment',
                             'Happy %', 'Angry %',
                             'Sad %', 'Surprise %', 'Fear %']],
                         use_container_width=True)

            # ── Download ──
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results as CSV",
                               csv, "results.csv", "text/csv")

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
            emoji = {"Positive": "😊",
                     "Negative": "😞",
                     "Neutral":  "😐"}.get(item['Sentiment'], "")
            st.markdown(
                f'<div class="history-item">{emoji} <b>{item["Sentiment"]}</b>'
                f' — {item["Text"]}</div>',
                unsafe_allow_html=True)
    else:
        st.info("No analysis done yet. Go to Single Review tab and analyze some text!")