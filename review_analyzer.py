"""
review_analyzer.py
-------------------
Turns hundreds of raw reviews into:
  1. Per-aspect scores (coffee, dessert, wifi, service, ambience, price, noise)
  2. A short, human-readable AI summary (extractive text summarization)
  3. A single "AI Recommendation Score" per cafe

This is the engine behind the "AI Review Analyzer" feature.
"""

import re
from collections import Counter

import pandas as pd

from sentiment import extract_aspect_sentiments, analyze_text, score_to_ten, _split_sentences

STOPWORDS = set("""
a an the this that these those is are was were be been being have has had do does did
will would shall should can could may might must to of in on at by for with about
against between into through during before after above below from up down out off over
under again further then once here there all any both each few more most other some such
no nor not only own same so than too very s t just don now i we you he she it they them
their his her its our your my me and or but if while as
""".split())


def keyword_summary(review_texts, top_n=6):
    """Very lightweight keyword extraction (word frequency) used to seed summaries."""
    words = re.findall(r"[a-zA-Z']+", " ".join(review_texts).lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 3]
    counts = Counter(words)
    return [w for w, _ in counts.most_common(top_n)]


def summarize_reviews(review_texts, max_sentences=3):
    """
    Extractive summarizer (word-frequency scoring, similar in spirit to a
    mini TextRank) — picks the most representative sentences across all
    reviews without needing a heavy transformer model.

    Swap in a call to a Hugging Face `summarization` pipeline
    (e.g. facebook/bart-large-cnn) here for an "advanced mode" if desired.
    """
    all_sentences = []
    for text in review_texts:
        all_sentences.extend(_split_sentences(text))

    if not all_sentences:
        return "No reviews available yet."

    word_freq = Counter()
    for sentence in all_sentences:
        for w in re.findall(r"[a-zA-Z']+", sentence.lower()):
            if w not in STOPWORDS:
                word_freq[w] += 1

    max_freq = max(word_freq.values()) if word_freq else 1
    for w in word_freq:
        word_freq[w] /= max_freq

    scored = []
    seen = set()
    for sentence in all_sentences:
        key = sentence.strip().lower()
        if key in seen or len(sentence.split()) < 4:
            continue
        seen.add(key)
        words = re.findall(r"[a-zA-Z']+", sentence.lower())
        score = sum(word_freq.get(w, 0) for w in words) / (len(words) + 1)
        scored.append((score, sentence.strip()))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in scored[:max_sentences]]
    return " ".join(top_sentences)


def analyze_cafe_reviews(cafe_row: pd.Series, reviews_df: pd.DataFrame) -> dict:
    """
    Given one cafe's metadata row and the full reviews dataframe,
    returns a full analysis dict combining stated metadata scores with
    review-derived sentiment, ready to render in the UI.
    """
    cafe_reviews = reviews_df[reviews_df["cafe_id"] == cafe_row["cafe_id"]]
    review_texts = cafe_reviews["review_text"].tolist()

    # Aggregate aspect sentiment across all reviews for this cafe
    aggregated = {}
    for text in review_texts:
        aspects = extract_aspect_sentiments(text)
        for aspect, score in aspects.items():
            aggregated.setdefault(aspect, []).append(score)

    aspect_scores_10 = {
        aspect: score_to_ten(sum(scores) / len(scores))
        for aspect, scores in aggregated.items()
    }

    avg_review_rating = round(cafe_reviews["rating"].mean(), 2) if len(cafe_reviews) else None
    summary = summarize_reviews(review_texts, max_sentences=3)
    keywords = keyword_summary(review_texts)

    weights = {
        "coffee_score": 0.20,
        "dessert_score": 0.15,
        "study_score": 0.15,
        "instagram_score": 0.15,
        "date_score": 0.15,
        "wifi_score": 0.20,
    }
    ai_score = round(sum(cafe_row[k] * w for k, w in weights.items()), 2)

    return {
        "cafe_id": cafe_row["cafe_id"],
        "name": cafe_row["name"],
        "num_reviews": len(review_texts),
        "avg_review_rating": avg_review_rating,
        "overall_rating": cafe_row["overall_rating"],
        "coffee_score": cafe_row["coffee_score"],
        "dessert_score": cafe_row["dessert_score"],
        "study_score": cafe_row["study_score"],
        "date_score": cafe_row["date_score"],
        "instagram_score": cafe_row["instagram_score"],
        "wifi_score": cafe_row["wifi_score"],
        "noise_level": cafe_row["noise_level"],
        "budget": cafe_row["budget"],
        "ai_recommendation_score": ai_score,
        "aspect_sentiment_10": aspect_scores_10,
        "ai_summary": summary,
        "keywords": keywords,
    }


if __name__ == "__main__":
    cafes = pd.read_csv("data/cafes.csv")
    reviews = pd.read_csv("data/reviews.csv")
    result = analyze_cafe_reviews(cafes.iloc[0], reviews)
    import json
    print(json.dumps(result, indent=2, default=str))
