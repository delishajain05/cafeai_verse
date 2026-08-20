"""
recommendation.py
------------------
Content-based cafe recommendation engine for CafeCompass AI.

Approach:
  1. Encode every cafe as a numeric feature vector (atmosphere one-hot,
     purpose one-hot, normalized scores, budget).
  2. Encode the user's stated preferences the same way.
  3. Rank cafes by cosine similarity to the user vector.
  4. (Optional) Use K-Nearest Neighbors as an alternative ranking method.

Hooks for scikit-learn:
  - sklearn.metrics.pairwise.cosine_similarity
  - sklearn.neighbors.NearestNeighbors
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

ATMOSPHERES = ["Minimal", "Heritage", "Rooftop", "Luxury", "Artsy", "Nature"]
PURPOSES = ["Study", "Date", "Friends", "Solo", "Work"]
BUDGET_MAP = {"Under 700": 1, "700 to 2000": 2, "Above 2000": 3}
BUDGET_RANGE_MAP = {"Under 700": "₹", "700 to 2000": "₹₹", "Above 2000": "₹₹₹"}

SCORE_COLS = ["coffee_score", "dessert_score", "study_score", "date_score",
              "instagram_score", "wifi_score"]


def _atmosphere_vector(atmosphere: str) -> list:
    return [1.0 if atmosphere == a else 0.0 for a in ATMOSPHERES]


def _purpose_vector(purposes) -> list:
    if isinstance(purposes, str):
        purposes = [p.strip() for p in purposes.replace(",", "|").split("|")]
    return [1.0 if p in purposes else 0.0 for p in PURPOSES]


def build_feature_matrix(cafes_df: pd.DataFrame):
    """
    Builds the full numeric feature matrix for all cafes.
    Returns feature_matrix (no distance column).
    """
    rows = []
    for _, cafe in cafes_df.iterrows():
        atm_vec = _atmosphere_vector(cafe["atmosphere"])
        purpose_vec = _purpose_vector(cafe["purpose_tags"])
        scores = [cafe[col] / 10.0 for col in SCORE_COLS]  # normalize 0-1
        budget_val = BUDGET_MAP.get(cafe.get("budget_label", "300 to 700"), 2) / 3.0
        rows.append(atm_vec + purpose_vec + scores + [budget_val])

    matrix = np.array(rows, dtype=float)
    return matrix


def build_user_vector(user_prefs: dict) -> np.ndarray:
    """
    user_prefs = {
        "atmosphere": "Minimal",
        "purpose": "Study",
        "budget": "700 to 2000",
        "priorities": {"coffee": 1.0, "dessert": 0.5, ...}  # optional 0-1 weights
    }
    """
    atm_vec = _atmosphere_vector(user_prefs.get("atmosphere", ""))
    purpose_vec = _purpose_vector([user_prefs.get("purpose", "")])

    priorities = user_prefs.get("priorities", {})
    default_priority_map = {
        "coffee_score": priorities.get("coffee", 0.8),
        "dessert_score": priorities.get("dessert", 0.6),
        "study_score": priorities.get("study", 0.5),
        "date_score": priorities.get("date", 0.5),
        "instagram_score": priorities.get("instagram", 0.5),
        "wifi_score": priorities.get("wifi", 0.5),
    }
    scores = [default_priority_map[col] for col in SCORE_COLS]

    budget_val = BUDGET_MAP.get(user_prefs.get("budget", "700 to 2000"), 2) / 3.0
    vec = np.array(atm_vec + purpose_vec + scores + [budget_val], dtype=float)
    return vec.reshape(1, -1)


def recommend_cafes(cafes_df: pd.DataFrame, user_prefs: dict, top_k: int = 5,
                     budget_label: str = None, method: str = "cosine") -> pd.DataFrame:
    """
    Returns the top_k cafes ranked by similarity to user_prefs.
    Optionally filter by budget_label (e.g. 'Under 700', '700 to 2000', 'Above 2000').
    method: "cosine" (default) or "knn"
    """
    working_df = cafes_df.copy()

    # Ensure budget_label column exists (map rupee symbols if not already mapped)
    if "budget_label" not in working_df.columns:
        symbol_to_label = {"₹": "Under 700", "₹₹": "700 to 2000", "₹₹₹": "Above 2000"}
        working_df["budget_label"] = working_df["budget"].map(symbol_to_label)

    if budget_label and budget_label != "Any":
        working_df = working_df[working_df["budget_label"] == budget_label]

    if working_df.empty:
        return working_df

    matrix = build_feature_matrix(working_df)
    user_vec = build_user_vector(user_prefs)

    if method == "knn":
        n_neighbors = min(top_k, len(working_df))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(matrix)
        distances, indices = knn.kneighbors(user_vec)
        similarities = 1 - distances[0]
        idx = indices[0]
    else:
        sims = cosine_similarity(user_vec, matrix)[0]
        idx = np.argsort(sims)[::-1][:top_k]
        similarities = sims[idx]

    result = working_df.iloc[idx].copy()
    result["match_score"] = np.round(similarities * 100, 1)
    return result.sort_values("match_score", ascending=False)


if __name__ == "__main__":
    cafes = pd.read_csv("data/cafes.csv")
    symbol_to_label = {"₹": "Under 700", "₹₹": "700 to 2000", "₹₹₹": "Above 2000"}
    cafes["budget_label"] = cafes["budget"].map(symbol_to_label)
    prefs = {
        "atmosphere": "Minimal",
        "purpose": "Study",
        "budget": "700 to 2000",
    }
    top = recommend_cafes(cafes, prefs, top_k=5)
    print(top[["name", "atmosphere", "purpose_tags", "match_score"]])
