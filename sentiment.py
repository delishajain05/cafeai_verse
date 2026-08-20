"""
sentiment.py
------------
Sentiment analysis and aspect-based opinion extraction for CafeCompass AI.

Two backends are supported:
  - "vader"      : fast, lexicon-based (default, no GPU/download needed)
  - "distilbert" : optional, more accurate transformer model
                   (only used if `transformers` + `torch` are installed
                   and ENABLE_DISTILBERT=True below)

Both backends expose the same interface: analyze_text(text) -> float in [-1, 1]
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ENABLE_DISTILBERT = False  # flip to True if transformers/torch are installed

_vader = SentimentIntensityAnalyzer()

_distilbert_pipeline = None
if ENABLE_DISTILBERT:
    try:
        from transformers import pipeline
        _distilbert_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    except Exception:
        _distilbert_pipeline = None


def analyze_text(text: str, backend: str = "vader") -> float:
    """Returns a compound sentiment score in [-1, 1]."""
    if backend == "distilbert" and _distilbert_pipeline is not None:
        result = _distilbert_pipeline(text[:512])[0]
        score = result["score"]
        return score if result["label"] == "POSITIVE" else -score
    # default: VADER
    return _vader.polarity_scores(text)["compound"]


# ---------------------------------------------------------------------------
# Aspect-based extraction
# ---------------------------------------------------------------------------

ASPECT_KEYWORDS = {
    "coffee": ["coffee", "espresso", "latte", "cappuccino", "brew", "americano", "filter coffee", "flat white"],
    "dessert": ["dessert", "cake", "pastry", "pastries", "sweet", "cheesecake", "brownie", "waffle"],
    "wifi": ["wifi", "wi-fi", "internet", "connection", "network"],
    "service": ["service", "staff", "server", "waiter", "waitress", "attentive", "hospitality"],
    "ambience": ["ambience", "ambiance", "decor", "vibe", "aesthetic", "interior", "photogenic", "instagram", "lighting"],
    "price": ["price", "expensive", "cheap", "pocket", "value", "cost", "budget", "affordable"],
    "noise": ["noise", "noisy", "quiet", "peaceful", "loud", "calm", "crowded"],
}


def _split_sentences(text: str):
    # lightweight sentence splitter (avoids needing nltk punkt download)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def extract_aspect_sentiments(review_text: str, backend: str = "vader") -> dict:
    """
    Scans a review for aspect-related sentences and returns per-aspect
    sentiment scores in [-1, 1]. Aspects not mentioned are omitted.
    """
    sentences = _split_sentences(review_text)
    aspect_scores = {aspect: [] for aspect in ASPECT_KEYWORDS}

    for sentence in sentences:
        lower = sentence.lower()
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                aspect_scores[aspect].append(analyze_text(sentence, backend=backend))

    return {
        aspect: sum(scores) / len(scores)
        for aspect, scores in aspect_scores.items()
        if scores
    }


def score_to_ten(compound_score: float) -> float:
    """Maps a VADER compound score [-1, 1] to a friendlier 0-10 scale."""
    return round((compound_score + 1) / 2 * 10, 1)


if __name__ == "__main__":
    sample = ("Came here with friends last evening. Good internet speed, I could work "
              "here for hours. The coffee here is absolutely rich and well brewed. "
              "The service was slow and we waited a long time.")
    print("Overall:", analyze_text(sample))
    print("Aspects:", extract_aspect_sentiments(sample))
