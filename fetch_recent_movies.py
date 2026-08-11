"""
Fetches recent movies (default: 2020 -> present) from TMDB and saves them
as data/new_movies.csv, in the same column shape as movies_metadata.csv,
so build_model.py can merge them into the recommender.

Requires TMDB_API_KEY in your .env file.

Run locally:
    python fetch_recent_movies.py
    python fetch_recent_movies.py --pages 40 --start-year 2022
"""

import argparse
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=1.5,  # 1.5s, 3s, 4.5s, 6s, 7.5s, 9s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (movie-recommender fetch script)"})
    return session


SESSION = make_session()


def _get_with_retry(url: str, params: dict, timeout: int = 20, attempts: int = 4):
    """Extra outer retry loop on top of the Session's built-in retries, for
    connection-level resets that happen before a response is ever received
    (common with antivirus/firewall SSL inspection on some Windows setups)."""
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            return SESSION.get(url, params=params, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last_err = e
            wait = attempt * 2
            print(f"    connection issue ({e.__class__.__name__}), retrying in {wait}s ({attempt}/{attempts})...")
            time.sleep(wait)
    raise last_err


def get_genre_map(api_key: str) -> dict:
    r = _get_with_retry(f"{TMDB_BASE}/genre/movie/list", {"api_key": api_key, "language": "en-US"})
    r.raise_for_status()
    return {g["id"]: g["name"] for g in r.json().get("genres", [])}


def discover_movies(api_key: str, start_year: int, end_year: int, pages: int) -> list:
    """Pulls popularity-sorted movies released between start_year and end_year."""
    results = []
    for page in range(1, pages + 1):
        try:
            r = _get_with_retry(
                f"{TMDB_BASE}/discover/movie",
                {
                    "api_key": api_key,
                    "language": "en-US",
                    "sort_by": "popularity.desc",
                    "include_adult": "false",
                    "primary_release_date.gte": f"{start_year}-01-01",
                    "primary_release_date.lte": f"{end_year}-12-31",
                    "page": page,
                },
                timeout=20,
            )
        except requests.exceptions.RequestException as e:
            print(f"  page {page}: gave up after retries ({e.__class__.__name__}), skipping this page.")
            continue
        if r.status_code != 200:
            print(f"  page {page}: TMDB error {r.status_code}, stopping.")
            break
        data = r.json()
        batch = data.get("results", [])
        if not batch:
            break
        results.extend(batch)
        total_pages = data.get("total_pages", pages)
        print(f"  page {page}/{min(pages, total_pages)} — {len(batch)} movies")
        if page >= total_pages:
            break
        time.sleep(0.3)  # be polite to the API
    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch recent movies from TMDB.")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--pages", type=int, default=40, help="~20 movies per page")
    parser.add_argument("--out", default=os.path.join("data", "new_movies.csv"))
    args = parser.parse_args()

    if not TMDB_API_KEY:
        raise SystemExit("TMDB_API_KEY not set. Add it to your .env file first.")

    print(f"Fetching genre list...")
    genre_map = get_genre_map(TMDB_API_KEY)

    print(f"Discovering movies from {args.start_year} to {args.end_year} ({args.pages} pages)...")
    raw = discover_movies(TMDB_API_KEY, args.start_year, args.end_year, args.pages)
    print(f"Fetched {len(raw)} movies.")

    rows = []
    for m in raw:
        genre_names = " ".join(genre_map.get(gid, "") for gid in m.get("genre_ids", []))
        rows.append(
            {
                "title": m.get("title", ""),
                "overview": m.get("overview", "") or "",
                "genres": genre_names,
                "tagline": "",  # not available from /discover; left blank
                "vote_average": m.get("vote_average"),
                "popularity": m.get("popularity"),
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip() != ""]
    df = df.drop_duplicates(subset=["title"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} movies to {args.out}")
    print("Now run: python build_model.py")


if __name__ == "__main__":
    main()
