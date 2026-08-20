"""
CafeCompass AI — Discover Your Perfect Cafe with AI
-----------------------------------------------------
Main Streamlit application. Run with:

    streamlit run app.py
"""

import os
import re

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from recommendation import recommend_cafes, ATMOSPHERES, PURPOSES
from review_analyzer import analyze_cafe_reviews

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

st.set_page_config(
    page_title="CafeCompass AI",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    cafes = pd.read_csv(os.path.join(DATA_DIR, "cafes.csv"))
    reviews = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))
    for col in ["pet_friendly", "outdoor_seating", "charging_sockets", "free_wifi", "open_now"]:
        cafes[col] = cafes[col].astype(str).str.lower().map({"true": True, "false": False})
    # Map rupee symbols to budget labels (new tiers: Under 700 / 700-2000 / Above 2000)
    symbol_to_label = {"₹": "Under 700", "₹₹": "700 to 2000", "₹₹₹": "Above 2000"}
    cafes["budget_label"] = cafes["budget"].map(symbol_to_label)
    # Normalize purpose_tags to use comma separator
    cafes["purpose_tags"] = cafes["purpose_tags"].str.replace("|", ",", regex=False)
    return cafes, reviews


@st.cache_data
def analyze_all_cafes(cafes_df: pd.DataFrame, reviews_df: pd.DataFrame):
    results = []
    for _, row in cafes_df.iterrows():
        results.append(analyze_cafe_reviews(row, reviews_df))
    return pd.DataFrame(results)


cafes_df, reviews_df = load_data()
analysis_df = analyze_all_cafes(cafes_df, reviews_df)
merged_df = cafes_df.merge(
    analysis_df[["cafe_id", "ai_recommendation_score", "ai_summary", "aspect_sentiment_10", "num_reviews", "avg_review_rating"]],
    on="cafe_id",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════
   FONTS
═══════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ═══════════════════════════════════════════════════════════════
   BASE
═══════════════════════════════════════════════════════════════ */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #fdf8f3 0%, #f9f1e8 40%, #fdf6ee 100%);
}

/* ═══════════════════════════════════════════════════════════════
   HERO / TITLE
═══════════════════════════════════════════════════════════════ */
.app-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(4rem, 10vw, 11rem);
    font-weight: 900;
    margin-bottom: 0;
    background: linear-gradient(135deg, #c8843a 0%, #6f4e37 35%, #a0522d 65%, #d4a855 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.5px;
    line-height: 1.1;
    background-size: 200% 200%;
    animation: titleShimmer 5s ease-in-out infinite alternate;
}

@keyframes titleShimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}

.app-tagline {
    font-size: 1.1rem;
    font-weight: 500;
    color: #7a6652;
    margin-top: 0.3rem;
    letter-spacing: 0.04rem;
    opacity: 0.9;
}

.title-accent {
    display: inline-block;
    width: 56px;
    height: 4px;
    background: linear-gradient(90deg, #c8843a, #6f4e37);
    border-radius: 2px;
    margin-top: 0.5rem;
    margin-bottom: 1.2rem;
    animation: accentGrow 0.8s ease-out forwards;
}

@keyframes accentGrow {
    from { width: 0; opacity: 0; }
    to   { width: 56px; opacity: 1; }
}

/* ═══════════════════════════════════════════════════════════════
   TAB BAR
═══════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(111, 78, 55, 0.06);
    padding: 6px 8px;
    border-radius: 14px;
    border: 1px solid rgba(200, 132, 58, 0.15);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 8px 22px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #7a6652 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.22s ease !important;
    letter-spacing: 0.02rem;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(200, 132, 58, 0.12) !important;
    color: #6f4e37 !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #c8843a, #6f4e37) !important;
    color: white !important;
    box-shadow: 0 3px 14px rgba(111, 78, 55, 0.28) !important;
}

.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ═══════════════════════════════════════════════════════════════
   METRIC TILES
═══════════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(253,248,243,0.92));
    border: 1px solid rgba(200, 132, 58, 0.2);
    border-radius: 14px;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 2px 12px rgba(111, 78, 55, 0.07);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
    backdrop-filter: blur(6px);
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(111, 78, 55, 0.15);
}

[data-testid="stMetricLabel"] {
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08rem !important;
    color: #a07850 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: #4a2e1a !important;
    font-family: 'Playfair Display', serif !important;
}

/* ═══════════════════════════════════════════════════════════════
   CAFE CARDS
═══════════════════════════════════════════════════════════════ */
.cafe-card {
    border: 1px solid rgba(200, 132, 58, 0.2);
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.1rem;
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(253,248,243,0.88));
    box-shadow: 0 2px 14px rgba(111, 78, 55, 0.07);
    transition: box-shadow 0.25s ease, transform 0.25s ease, border-color 0.25s ease;
    backdrop-filter: blur(8px);
    position: relative;
    overflow: hidden;
}

.cafe-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, #c8843a, #6f4e37);
    border-radius: 18px 0 0 18px;
    opacity: 0;
    transition: opacity 0.25s ease;
}

.cafe-card:hover {
    box-shadow: 0 10px 36px rgba(111, 78, 55, 0.18);
    transform: translateY(-4px);
    border-color: rgba(200, 132, 58, 0.48);
}

.cafe-card:hover::before { opacity: 1; }

/* ═══════════════════════════════════════════════════════════════
   SCORE / BADGE PILLS
═══════════════════════════════════════════════════════════════ */
.score-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    background: linear-gradient(135deg, #c8843a, #6f4e37);
    color: white;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 6px;
    letter-spacing: 0.02rem;
    box-shadow: 0 2px 8px rgba(111,78,55,0.25);
}

.new-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 6px;
    background: linear-gradient(90deg, #e8f4fd, #ddeeff);
    color: #1a6fb5;
    font-size: 0.72rem;
    font-weight: 700;
    margin-left: 8px;
    vertical-align: middle;
    letter-spacing: 0.02rem;
}

/* ═══════════════════════════════════════════════════════════════
   SUBHEADERS & SECTION TITLES
═══════════════════════════════════════════════════════════════ */
h2, .stSubheader {
    font-family: 'Playfair Display', serif !important;
    color: #4a2e1a !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}

h3 { font-weight: 700 !important; color: #5c3820 !important; }

/* ═══════════════════════════════════════════════════════════════
   FORM INPUTS
═══════════════════════════════════════════════════════════════ */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid rgba(200, 132, 58, 0.3) !important;
    background: rgba(255,255,255,0.92) !important;
    color: #3d2510 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #c8843a !important;
    box-shadow: 0 0 0 3px rgba(200, 132, 58, 0.15) !important;
}

.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1.5px solid rgba(200, 132, 58, 0.3) !important;
    background: rgba(255,255,255,0.92) !important;
    transition: border-color 0.2s ease !important;
}

.stSelectbox > div > div:focus-within {
    border-color: #c8843a !important;
    box-shadow: 0 0 0 3px rgba(200, 132, 58, 0.15) !important;
}

[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #c8843a, #6f4e37) !important;
}

[data-testid="stSlider"] > div > div > div > div > div {
    background: white !important;
    border: 2px solid #c8843a !important;
    box-shadow: 0 2px 8px rgba(200, 132, 58, 0.3) !important;
}

.stCheckbox label {
    font-weight: 500 !important;
    color: #5c3820 !important;
    font-size: 0.88rem !important;
}

/* ═══════════════════════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════════════════════ */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
    border: 1.5px solid rgba(200,132,58,0.35) !important;
    color: #6f4e37 !important;
    background: rgba(255,255,255,0.88) !important;
}

.stButton > button:hover {
    background: rgba(200,132,58,0.1) !important;
    border-color: #c8843a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(111,78,55,0.15) !important;
}

.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #c8843a 0%, #6f4e37 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 3px 14px rgba(111, 78, 55, 0.30) !important;
    font-weight: 700 !important;
    letter-spacing: 0.02rem !important;
}

.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #b5752f 0%, #5d3e2b 100%) !important;
    box-shadow: 0 6px 20px rgba(111, 78, 55, 0.40) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #c8843a 0%, #6f4e37 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03rem !important;
    box-shadow: 0 4px 16px rgba(111, 78, 55, 0.30) !important;
    transition: all 0.22s ease !important;
}

[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #b5752f 0%, #5d3e2b 100%) !important;
    box-shadow: 0 8px 24px rgba(111, 78, 55, 0.42) !important;
    transform: translateY(-2px) !important;
}

/* ═══════════════════════════════════════════════════════════════
   EXPANDERS
═══════════════════════════════════════════════════════════════ */
details[data-testid="stExpander"] {
    border: 1px solid rgba(200, 132, 58, 0.22) !important;
    border-radius: 12px !important;
    background: rgba(253, 248, 243, 0.72) !important;
    transition: box-shadow 0.2s ease;
}

details[data-testid="stExpander"]:hover {
    box-shadow: 0 4px 18px rgba(111,78,55,0.10);
}

details[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #6f4e37 !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1rem !important;
}

/* ═══════════════════════════════════════════════════════════════
   ALERT BOXES
═══════════════════════════════════════════════════════════════ */
[data-testid="stInfo"] {
    background: linear-gradient(135deg, rgba(200,132,58,0.08), rgba(111,78,55,0.05)) !important;
    border: 1px solid rgba(200,132,58,0.22) !important;
    border-radius: 12px !important;
    color: #5c3820 !important;
}

[data-testid="stSuccess"] {
    background: rgba(56, 142, 60, 0.07) !important;
    border: 1px solid rgba(56, 142, 60, 0.28) !important;
    border-radius: 12px !important;
}

[data-testid="stWarning"] {
    background: rgba(255, 167, 38, 0.08) !important;
    border-radius: 12px !important;
}

/* ═══════════════════════════════════════════════════════════════
   PLOTLY CHARTS
═══════════════════════════════════════════════════════════════ */
[data-testid="stPlotlyChart"] {
    border-radius: 14px !important;
    overflow: hidden;
    box-shadow: 0 2px 14px rgba(111,78,55,0.08);
    border: 1px solid rgba(200,132,58,0.12);
    transition: box-shadow 0.22s ease;
}

[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 6px 26px rgba(111,78,55,0.15);
}

/* ═══════════════════════════════════════════════════════════════
   DIVIDERS
═══════════════════════════════════════════════════════════════ */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(200,132,58,0.32), transparent) !important;
    margin: 1.2rem 0 !important;
}

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════════════════ */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fdf5ec 0%, #f8ede0 100%);
    border-right: 1px solid rgba(200, 132, 58, 0.15);
}

div[data-testid="stSidebar"] .stButton > button {
    border-radius: 999px !important;
    padding: 4px 14px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    border: 1.5px solid #c8843a !important;
    background: transparent !important;
    color: #7a6652 !important;
    margin: 2px 2px !important;
    transition: all 0.18s ease !important;
    line-height: 1.4 !important;
    min-height: unset !important;
    height: auto !important;
}

div[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(200,132,58,0.12) !important;
    color: #6f4e37 !important;
    border-color: #6f4e37 !important;
    transform: none !important;
    box-shadow: none !important;
}

div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #c8843a, #6f4e37) !important;
    color: white !important;
    border-color: transparent !important;
}

/* ═══════════════════════════════════════════════════════════════
   CAPTION / FOOTER
═══════════════════════════════════════════════════════════════ */
.stCaption, [data-testid="stCaptionContainer"] p {
    color: #9a7a5e !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.01rem;
}

/* ═══════════════════════════════════════════════════════════════
   CUSTOM SCROLLBAR
═══════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(200,132,58,0.05); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #c8843a, #6f4e37); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6f4e37; }

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="app-title">CafeCompass AI</p>', unsafe_allow_html=True)
st.markdown('<div class="title-accent"></div>', unsafe_allow_html=True)
st.markdown('<p class="app-tagline">Your AI-powered guide to Jaipur\'s finest cafes — matched to your taste, powered by review intelligence.</p>', unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------------------
# Helper render functions
# ---------------------------------------------------------------------------

def radar_chart(scores: dict, title: str):
    categories = list(scores.keys())
    values = list(scores.values())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]],
                                   fill="toself", name=title, line_color="#6f4e37"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False, height=320, margin=dict(l=30, r=30, t=30, b=30),
    )
    return fig


def render_cafe_card(row, show_match=False):
    name = row["name"]
    with st.container():
        st.markdown('<div class="cafe-card">', unsafe_allow_html=True)
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"### {name}")
            st.caption(f"{row['area']}, {row['location']} · {row['budget']} · {row.get('atmosphere', '')} atmosphere · Noise: {row['noise_level']}")
            tags = f"{row['purpose_tags'].replace(',', ', ')} · {row['budget_label']}"
            st.write(tags)
        with cols[1]:
            if show_match and "match_score" in row:
                st.metric("Match", f"{row['match_score']}%")
            st.metric("AI Score", f"{row['ai_recommendation_score']}/10")

        score_cols = st.columns(6)
        labels = [("Coffee", "coffee_score"), ("Dessert", "dessert_score"),
                  ("Study", "study_score"), ("Date", "date_score"),
                  ("Instagram", "instagram_score"), ("Wi-Fi", "wifi_score")]
        for col, (label, key) in zip(score_cols, labels):
            col.metric(label, f"{row[key]}/10")

        with st.expander("AI Review Summary"):
            st.write(row.get("ai_summary", "No summary available."))
            st.caption(f"Based on {row.get('num_reviews', 0)} reviews · Avg reviewer rating: {row.get('avg_review_rating', 'N/A')}/5")

        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_dashboard, tab_recommend, tab_analyzer = st.tabs(
    ["Dashboard", "Get Recommendation", "Review Analyzer"]
)

# --- Dashboard ---------------------------------------------------------------
with tab_dashboard:
    st.subheader("Explore Jaipur's Cafe Scene")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cafes", len(merged_df))
    c2.metric("Total Reviews Analyzed", int(merged_df["num_reviews"].sum()))
    c3.metric("Avg AI Score", f"{merged_df['ai_recommendation_score'].mean():.1f}/10")
    c4.metric("Avg Overall Rating", f"{merged_df['overall_rating'].mean():.1f}/5")

    st.write("")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Trending Cafes (by AI Score)**")
        top_trending = merged_df.sort_values("ai_recommendation_score", ascending=False).head(8).copy()
        fig = px.bar(top_trending, x="ai_recommendation_score", y="name", orientation="h",
                     color="ai_recommendation_score", color_continuous_scale="Oranges",
                     labels={"ai_recommendation_score": "AI Score", "name": ""})
        fig.update_layout(yaxis=dict(autorange="reversed"), height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown("**Most Instagrammable Cafes**")
        top_insta = merged_df.sort_values("instagram_score", ascending=False).head(8).copy()
        fig = px.bar(top_insta, x="instagram_score", y="name", orientation="h",
                     color="instagram_score", color_continuous_scale="Pinkyl",
                     labels={"instagram_score": "Instagram Score", "name": ""})
        fig.update_layout(yaxis=dict(autorange="reversed"), height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        st.markdown("**Best Study Cafes**")
        for _, r in merged_df.sort_values("study_score", ascending=False).head(4).iterrows():
            st.write(f"**{r['name']}** — {r['area']} · Study score {r['study_score']}/10")

        st.markdown("**Highest-Rated Date Cafes**")
        for _, r in merged_df.sort_values("date_score", ascending=False).head(4).iterrows():
            st.write(f"**{r['name']}** — {r['area']} · Date score {r['date_score']}/10")

    with colD:
        st.markdown("**Best Coffee Spots**")
        for _, r in merged_df.sort_values("coffee_score", ascending=False).head(4).iterrows():
            st.write(f"**{r['name']}** — {r['area']} · Coffee score {r['coffee_score']}/10")

        st.markdown("**Budget-Friendly Picks**")
        for _, r in merged_df[merged_df["budget"] == "₹"].sort_values("ai_recommendation_score", ascending=False).head(4).iterrows():
            st.write(f"**{r['name']}** — {r['area']} · AI score {r['ai_recommendation_score']}/10")

# --- Recommendation ----------------------------------------------------------
with tab_recommend:
    st.subheader("Tell us your preferences — we will find your cafe")

    with st.form("recommend_form"):
        f1, f2 = st.columns(2)
        location = f1.text_input("Location", value="Jaipur")

        # Budget text input
        budget_input = f2.text_input("Budget (per person in Rs)", value="500",
                                      help="Enter your budget per person. e.g. 200, 500, 1000")

        f4, f5 = st.columns(2)
        atmosphere = f4.selectbox("Atmosphere", ATMOSPHERES)
        purpose = f5.selectbox("Purpose", PURPOSES)

        # Seating Preference (replaces Vibe section)
        st.markdown("**Seating Preference**")
        s1, s2, s3 = st.columns(3)
        prefer_outdoor = s1.checkbox("Outdoor Seating")
        prefer_quiet = s2.checkbox("Quiet / Low Noise")
        prefer_wifi = s3.checkbox("Free Wi-Fi Required")

        st.markdown("**Fine-tune what matters most to you** (optional)")
        p1, p2, p3 = st.columns(3)
        w_coffee = p1.slider("Coffee importance", 0.0, 1.0, 0.8)
        w_dessert = p1.slider("Dessert importance", 0.0, 1.0, 0.6)
        w_study = p2.slider("Study-friendliness", 0.0, 1.0, 0.5)
        w_date = p2.slider("Date-friendliness", 0.0, 1.0, 0.5)
        w_insta = p3.slider("Instagram-worthiness", 0.0, 1.0, 0.5)
        w_wifi = p3.slider("Wi-Fi quality", 0.0, 1.0, 0.5)

        submitted = st.form_submit_button("Find My Perfect Cafe", use_container_width=True)

    if submitted:
        # Parse budget text input
        try:
            budget_amount = int(re.sub(r"[^\d]", "", budget_input))
        except (ValueError, TypeError):
            budget_amount = 700

        if budget_amount <= 700:
            budget_label = "Under 700"
        elif budget_amount <= 2000:
            budget_label = "700 to 2000"
        else:
            budget_label = "Above 2000"

        user_prefs = {
            "atmosphere": atmosphere,
            "purpose": purpose,
            "budget": budget_label,
            "priorities": {
                "coffee": w_coffee, "dessert": w_dessert, "study": w_study,
                "date": w_date, "instagram": w_insta, "wifi": w_wifi,
            },
        }
        st.info(f"Searching for cafes with budget: {budget_label} (Rs {budget_amount})", icon=None)

        # Apply seating preference pre-filters
        filtered_cafes = cafes_df.copy()
        if prefer_outdoor:
            filtered_cafes = filtered_cafes[filtered_cafes["outdoor_seating"] == True]
        if prefer_quiet:
            filtered_cafes = filtered_cafes[filtered_cafes["noise_level"] == "Low"]
        if prefer_wifi:
            filtered_cafes = filtered_cafes[filtered_cafes["free_wifi"] == True]

        top_matches = recommend_cafes(filtered_cafes, user_prefs, top_k=5, budget_label=budget_label)

        if top_matches.empty:
            st.warning("No cafes found with those preferences. Try adjusting your filters.")
        else:
            merged_matches = top_matches.merge(
                analysis_df[["cafe_id", "ai_recommendation_score", "ai_summary", "num_reviews", "avg_review_rating"]],
                on="cafe_id",
            )
            st.success(f"Found {len(merged_matches)} great matches for you!")
            for _, row in merged_matches.iterrows():
                render_cafe_card(row, show_match=True)

# --- Review Analyzer ----------------------------------------------------------
with tab_analyzer:
    st.subheader("AI Review Analyzer")
    st.caption("Instead of reading hundreds of reviews, get an instant AI-generated summary and score card.")

    display_names = merged_df["name"].tolist()
    cafe_display = st.selectbox("Select a cafe", display_names)
    selected = merged_df[merged_df["name"] == cafe_display].iloc[0]
    analysis = analysis_df[analysis_df["cafe_id"] == selected["cafe_id"]].iloc[0]

    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"## {selected['name']}")
        st.caption(f"{selected['area']}, {selected['location']} · {selected['budget']} · {selected.get('atmosphere', '')} atmosphere")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Rating", f"{selected['overall_rating']}/5")
        m2.metric("AI Score", f"{selected['ai_recommendation_score']}/10")
        m3.metric("Noise Level", selected["noise_level"])
        m4.metric("Reviews Analyzed", int(analysis["num_reviews"]))

        st.markdown("#### AI Summary")
        st.info(analysis["ai_summary"])

        st.markdown("#### Detailed Scores")
        score_grid = st.columns(3)
        detail_labels = [
            ("Coffee Quality", "coffee_score"), ("Dessert Rating", "dessert_score"),
            ("Study Friendly", "study_score"), ("Date Friendly", "date_score"),
            ("Instagram Score", "instagram_score"), ("Wi-Fi Quality", "wifi_score"),
        ]
        for i, (label, key) in enumerate(detail_labels):
            score_grid[i % 3].metric(label, f"{selected[key]}/10")

    with right:
        st.markdown("#### Score Radar")
        radar_scores = {
            "Coffee": selected["coffee_score"], "Dessert": selected["dessert_score"],
            "Study": selected["study_score"], "Date": selected["date_score"],
            "Instagram": selected["instagram_score"], "Wi-Fi": selected["wifi_score"],
        }
        st.plotly_chart(radar_chart(radar_scores, selected["name"]), use_container_width=True)

    st.markdown("#### Aspect Sentiment (from raw review text)")
    aspect_sent = analysis["aspect_sentiment_10"]
    if aspect_sent:
        aspect_df = pd.DataFrame(list(aspect_sent.items()), columns=["Aspect", "Score (0-10)"])
        fig = px.bar(aspect_df, x="Score (0-10)", y="Aspect", orientation="h", range_x=[0, 10],
                     color="Score (0-10)", color_continuous_scale="Oranges")
        fig.update_layout(coloraxis_showscale=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"Read all {len(reviews_df[reviews_df['cafe_id'] == selected['cafe_id']])} raw reviews"):
        cafe_reviews = reviews_df[reviews_df["cafe_id"] == selected["cafe_id"]]
        for _, r in cafe_reviews.iterrows():
            stars = int(r["rating"])
            st.write(f"[{stars}/5] — {r['review_text']}")
            st.divider()

st.write("")
st.caption("CafeVerse AI · Built with Streamlit, scikit-learn, and VADER sentiment analysis · Data for Jaipur")
