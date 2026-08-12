"""
Filters a full TMDB movie dump (e.g. TMDB_all_movies.csv, ~1.2M rows) down to
a well-known, appropriately-sized subset, and reshapes it into the schema
build_model.py expects: title, overview, genres, tagline, vote_average,
popularity.

The full dump is too large to use directly (1.2M+ rows would blow past
free-tier memory/storage limits on GitHub and Streamlit Cloud), so this
keeps only Released movies with enough votes to be "real" entries.

Run locally:
    python prepare_dataset.py --source data/TMDB_all_movies.csv
    python prepare_dataset.py --source data/TMDB_all_movies.csv --min-votes 20
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to the full TMDB dump CSV")
    parser.add_argument("--out", default=os.path.join("data", "movies_metadata.csv"))
    parser.add_argument("--min-votes", type=int, default=10, help="Minimum vote_count to keep a movie")
    parser.add_argument(
        "--boost-languages",
        default="hi",
        help="Comma-separated ISO 639-1 codes (e.g. 'hi' for Hindi/Bollywood) that get a lower "
        "vote threshold, since non-Hollywood industries are systematically under-voted on TMDB "
        "relative to their actual popularity. Empty string disables boosting.",
    )
    parser.add_argument("--boost-min-votes", type=int, default=3, help="Vote threshold for boosted languages")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    boost_langs = {c.strip() for c in args.boost_languages.split(",") if c.strip()}
    cols_needed = [
        "title", "overview", "genres", "tagline", "vote_average", "popularity",
        "vote_count", "status", "original_language",
    ]

    print(f"Reading {args.source} in chunks of {args.chunksize} ...")
    if boost_langs:
        print(f"Boosting languages {boost_langs} down to vote_count >= {args.boost_min_votes}")
    kept_chunks = []
    total_seen = 0
    for chunk in pd.read_csv(args.source, usecols=cols_needed, chunksize=args.chunksize, low_memory=False):
        total_seen += len(chunk)
        chunk = chunk[chunk["status"] == "Released"]
        votes = chunk["vote_count"].fillna(0)
        is_boosted = chunk["original_language"].isin(boost_langs)
        keep_mask = (votes >= args.min_votes) | (is_boosted & (votes >= args.boost_min_votes))
        chunk = chunk[keep_mask]
        chunk = chunk.dropna(subset=["title"])
        chunk = chunk[chunk["title"].astype(str).str.strip() != ""]
        if len(chunk):
            kept_chunks.append(chunk)
        print(f"  scanned {total_seen:,} rows so far, kept {sum(len(c) for c in kept_chunks):,}")

    df = pd.concat(kept_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["title"])

    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")
    df["genres"] = (
        df["genres"]
        .fillna("")
        .apply(lambda s: " ".join(part.strip() for part in str(s).split(",") if part.strip()))
    )

    df = df[["title", "overview", "genres", "tagline", "vote_average", "popularity"]]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} movies to {args.out}")
    print("Now run: python build_model.py")


if __name__ == "__main__":
    main()
