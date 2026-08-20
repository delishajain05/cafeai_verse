# ☕ CafeCompass AI

**Discover Your Perfect Cafe with AI**

An AI-powered platform that recommends the perfect cafe based on your vibe and
analyzes hundreds of customer reviews to help you decide where to go.

This repo is a fully working prototype: sample data for **18 cafes across
Jaipur** and **~185 synthetic reviews**, a content-based recommendation
engine, and an aspect-based sentiment/review-summarization pipeline — all
wired up in a Streamlit app.

---

## 🚀 Quick Start

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate the sample dataset
python data/generate_data.py

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🗂️ Project Structure

```
CafeCompassAI/
│
├── app.py                 # Streamlit UI — dashboard, recommender, analyzer, filters, chatbot
├── recommendation.py       # Content-based filtering + cosine similarity / KNN recommender
├── review_analyzer.py      # Aggregates reviews per cafe → scores, AI summary, keywords
├── sentiment.py             # VADER sentiment + aspect-based opinion extraction
├── data/
│   ├── cafes.csv           # 18 sample Jaipur cafes with metadata + scores
│   ├── reviews.csv         # ~185 synthetic customer reviews
│   └── generate_data.py    # Script to regenerate/extend the dataset
├── models/                  # (empty) drop any pickled/trained models here
├── images/                  # (empty) drop cafe photos here if you add a gallery
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

### 1. Personalized Cafe Recommendation (`recommendation.py`)
Each cafe is encoded as a numeric vector: vibe (one-hot), purpose tags
(one-hot), normalized 0–1 scores (coffee, dessert, study, date, Instagram,
Wi-Fi), budget tier, and distance. The user's stated preferences are encoded
the same way, and cafes are ranked by **cosine similarity**
(`sklearn.metrics.pairwise.cosine_similarity`). A **K-Nearest Neighbors**
mode (`method="knn"`) is also available as an alternative ranking strategy.

### 2. AI Review Analyzer (`sentiment.py` + `review_analyzer.py`)
- **Sentiment analysis** uses `VADER` (`vaderSentiment`) by default — fast,
  lexicon-based, no downloads required. A `DistilBERT` backend hook is
  included in `sentiment.py` (`ENABLE_DISTILBERT = True`) for higher accuracy
  if you install `transformers` + `torch`.
- **Aspect extraction** scans each review sentence for keywords related to
  coffee, desserts, Wi-Fi, service, ambience, price, and noise, and scores
  the sentiment of just those sentences — so you get a "Coffee: 8.5/10"
  style breakdown instead of one blended number.
- **Text summarization** uses a lightweight extractive, word-frequency
  scoring algorithm (similar in spirit to TextRank) to compress dozens of
  reviews into a 2–3 sentence AI summary — no heavy transformer model
  required, though you can swap in a Hugging Face `summarization` pipeline
  (e.g. `facebook/bart-large-cnn`) for an "advanced mode".

### 3. AI Recommendation Score
A weighted blend of coffee, dessert, study-friendliness, Instagram-worthiness,
date-friendliness, and Wi-Fi quality — giving a single "at a glance" score
instead of just a star rating.

---

## 🌟 Features Implemented

- ✅ Personalized recommendation form (location, budget, vibe, purpose, distance, priority sliders)
- ✅ AI Review Analyzer with score cards, radar chart, and generated summary
- ✅ Smart filters sidebar (budget, open now, pet friendly, rooftop, outdoor seating,
  charging sockets, free Wi-Fi, best desserts/coffee, quiet, student/couple-friendly)
- ✅ Dashboard: trending cafes, top study/date/coffee/dessert/budget picks, Instagram leaderboard
- ✅ Working prototype chatbot ("Find a quiet cafe under ₹500 for studying")

## 🔭 Future Enhancements (roadmap, not yet implemented)

- 🌦️ Weather-based recommendations
- 👗 Match cafes to outfit colors/aesthetics
- 🗺️ Interactive map (Google Maps Places API — swap in real data via `data/generate_data.py`)
- ⭐ Save favorite cafes (needs persistent storage / user accounts)
- 📅 AI-generated cafe-hopping itinerary for a full day

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| Recommendation | scikit-learn (cosine similarity, KNN) |
| NLP | Custom keyword/aspect extraction |
| Sentiment Analysis | VADER (default) / DistilBERT (optional) |
| Database | CSV (swap for SQLite easily — see note below) |
| Visualization | Plotly |
| Data Processing | Pandas |

> **Note on SQLite:** the prototype ships with CSV files for simplicity and
> easy editing. To move to SQLite, load `data/cafes.csv` and `data/reviews.csv`
> into a `cafes` and `reviews` table with `pandas.DataFrame.to_sql`, then
> replace the `pd.read_csv(...)` calls in `app.py` with `pd.read_sql(...)`.

---

## ➕ Adding Real Data

Replace the contents of `data/cafes.csv` / `data/reviews.csv` with real data
(e.g. pulled from the Google Maps Places API) — just keep the same column
names, and everything downstream (recommender, analyzer, dashboard) will
work unchanged. `data/generate_data.py` shows the exact schema expected.
