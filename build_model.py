"""
Builds the TF-IDF content-based movie recommender model.

Reads data/movies_metadata.csv, cleans it, engineers a 'tags' field from
overview + genres + tagline, fits a TF-IDF vectorizer, and pickles
everything the API needs at runtime into app/model/.

Run once locally:  python build_model.py
"""

import ast
import os
import pickle
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = os.path.join("data", "movies_metadata.csv")
MODEL_DIR = os.path.join("app", "model")

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def parse_genres(raw: str) -> str:
    """Turn the stringified list-of-dicts genres column into a plain string."""
    try:
        items = ast.literal_eval(raw)
        return " ".join(i["name"] for i in items)
    except (ValueError, SyntaxError):
        return ""


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    words = [LEMMATIZER.lemmatize(w) for w in words]
    return " ".join(words)


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    df = df.drop_duplicates().reset_index(drop=True)
    df = df[["title", "overview", "genres", "tagline", "vote_average", "popularity"]]
    df = df.dropna(subset=["title"])
    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")
    df["genres"] = df["genres"].apply(parse_genres)

    df["tags"] = df["overview"] + " " + df["genres"] + " " + df["tagline"]
    df["tags"] = df["tags"].apply(preprocess_text)

    df = df.reset_index(drop=True)
    # Keep the last occurrence's index for duplicate titles so lookups are stable
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

    print(f"{len(df)} movies after cleaning. Fitting TF-IDF ...")
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["tags"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    df.to_pickle(os.path.join(MODEL_DIR, "df.pkl"))
    with open(os.path.join(MODEL_DIR, "indices.pkl"), "wb") as f:
        pickle.dump(indices, f)
    with open(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(os.path.join(MODEL_DIR, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    print(f"Saved model artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
