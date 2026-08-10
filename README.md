# 🎬 Movie Recommender

A content-based movie recommender using TF-IDF over each movie's overview,
genres, and tagline. Built with **Streamlit** (UI) and **FastAPI** (optional
standalone API), on top of ["The Movies Dataset"](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

## Project structure

```
movie-recommender/
├── app/
│   ├── main.py            # FastAPI backend (also importable for the Streamlit app)
│   ├── streamlit_app.py   # Streamlit UI — the main entry point
│   └── model/             # Pre-built TF-IDF model artifacts (committed)
├── data/
│   └── README.md          # Where to get the dataset if you want to retrain
├── build_model.py         # Rebuilds the model from movies_metadata.csv
├── requirements.txt
├── .env.example
└── README.md
```

## Run it locally

```bash
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Optional: enables movie posters
cp .env.example .env
# then edit .env and add your TMDB API key (free at https://www.themoviedb.org/settings/api)

cd app
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

The pre-built model in `app/model/` is included, so it works immediately —
no need to download the dataset unless you want to retrain (see
`data/README.md`).

### Running the standalone API (optional)

```bash
cd app
uvicorn main:app --reload
```

Docs at http://localhost:8000/docs.

## Deploy for free — Streamlit Community Cloud

This is the easiest way to get a public link, and it's what the UI in this
repo is built for.

1. Push this repo to GitHub (see below).
2. Go to https://share.streamlit.io → **New app**.
3. Pick your repo, set:
   - **Main file path:** `app/streamlit_app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   TMDB_API_KEY = "your_tmdb_api_key_here"
   ```
   (Optional — the app works without it, just without posters.)
5. Click **Deploy**. You'll get a public `*.streamlit.app` URL.

## Push this project to GitHub

```bash
cd movie-recommender
git init
git add .
git commit -m "Movie recommender: TF-IDF model + Streamlit UI + FastAPI backend"
git branch -M main
git remote add origin https://github.com/<your-username>/movie-recommender.git
git push -u origin main
```

**Important:** never commit your real `.env` file or API key — `.gitignore`
already excludes `.env`. Only `.env.example` (with a placeholder) is tracked.

## How the recommender works

1. `build_model.py` cleans the dataset and combines each movie's `overview`,
   `genres`, and `tagline` into one `tags` field.
2. A `TfidfVectorizer` (unigrams + bigrams, 50k features) turns every movie's
   `tags` into a vector.
3. Recommending for a title = cosine similarity between that movie's vector
   and all others, sorted descending, top N returned.
