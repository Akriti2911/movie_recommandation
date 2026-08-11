"""
Streamlit UI for the TF-IDF movie recommender.

Runs the recommendation logic in-process (no separate FastAPI server needed
for deployment on Streamlit Community Cloud), reusing the same pickled
model artifacts and TMDB helper functions from main.py.
"""

import asyncio

import streamlit as st

from main import (
    COMMON_GENRES,
    TMDB_API_KEY,
    attach_tmdb_card_by_title,
    get_movie_row,
    load_pickles,
    tfidf_recommend_titles,
    tmdb_cards_from_results,
    tmdb_get,
)

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b0d17 0%, #14152b 100%);
    }
    .hero {
        padding: 3rem 1rem 2rem 1rem;
        text-align: center;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b6e, #ff9a5a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: #9aa0b4;
        font-size: 1.05rem;
    }
    .movie-card {
        background: #1a1c30;
        border-radius: 14px;
        overflow: hidden;
        transition: transform 0.15s ease;
        border: 1px solid #262844;
        height: 100%;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        border-color: #ff4b6e;
    }
    .movie-poster {
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
        background: #262844;
    }
    .movie-poster-placeholder {
        width: 100%;
        aspect-ratio: 2/3;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #262844;
        color: #565a78;
        font-size: 2.5rem;
    }
    .movie-info {
        padding: 0.7rem 0.8rem 0.9rem 0.8rem;
    }
    .movie-title {
        color: #f0f1f7;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
        line-height: 1.25;
        min-height: 2.4em;
    }
    .movie-score {
        display: inline-block;
        background: rgba(255, 75, 110, 0.15);
        color: #ff8095;
        font-size: 0.75rem;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
    }
    div[data-testid="stTextInput"] input {
        background: #1a1c30;
        color: #f0f1f7;
        border: 1px solid #262844;
        border-radius: 10px;
        padding: 0.7rem 1rem;
    }
    .stButton button {
        background: linear-gradient(90deg, #ff4b6e, #ff9a5a);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def init_model():
    load_pickles()
    return True


init_model()


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@st.cache_data(show_spinner=False)
def fetch_poster(title: str):
    if not TMDB_API_KEY:
        return None
    try:
        card = run_async(attach_tmdb_card_by_title(title))
        return card.poster_url if card else None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_trending(limit: int = 10):
    if not TMDB_API_KEY:
        return []
    try:
        data = run_async(tmdb_get("/trending/movie/day", {"language": "en-US"}))
        cards = run_async(tmdb_cards_from_results(data.get("results", []), limit=limit))
        return cards
    except Exception:
        return []


def render_movie_card(title: str, poster_url, badge: str, key_prefix: str):
    poster_html = (
        f'<img class="movie-poster" src="{poster_url}">'
        if poster_url
        else '<div class="movie-poster-placeholder">🎬</div>'
    )
    st.markdown(
        f"""
        <div class="movie-card">
            {poster_html}
            <div class="movie-info">
                <div class="movie-title">{title}</div>
                <span class="movie-score">{badge}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    row = get_movie_row(title)
    if row and (row.get("overview") or row.get("tagline")):
        with st.expander("Details", expanded=False):
            if row.get("tagline"):
                st.markdown(f"*{row['tagline']}*")
            if row.get("genres"):
                st.caption(row["genres"])
            if row.get("overview"):
                st.write(row["overview"])
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)


# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
        <h1>🎬 CineMatch</h1>
        <p>Find your next favorite movie — content-based recommendations powered by TF-IDF.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not TMDB_API_KEY:
    st.info(
        "No TMDB_API_KEY set — recommendations will still work, just without posters. "
        "Add one in `.env` (local) or Secrets (Streamlit Cloud) to enable posters.",
        icon="ℹ️",
    )

# ---------- Trending Now ----------
if TMDB_API_KEY:
    trending = fetch_trending(limit=10)
    if trending:
        st.markdown("### 🔥 Trending now")
        cols = st.columns(5)
        for i, card in enumerate(trending):
            with cols[i % 5]:
                render_movie_card(
                    card.title,
                    card.poster_url,
                    f"⭐ {card.vote_average:.1f}" if card.vote_average else "",
                    key_prefix=f"trend-{i}",
                )
        st.markdown("---")

# ---------- Search ----------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    query = st.text_input(
        "Type a movie title",
        placeholder="e.g. Toy Story, The Dark Knight, Inception...",
        label_visibility="collapsed",
    )
    genre_choice = st.selectbox("Filter by genre (optional)", ["Any genre"] + COMMON_GENRES)
    top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
    go = st.button("🔍 Find similar movies", type="primary", use_container_width=True)

if go and query:
    genre_filter = None if genre_choice == "Any genre" else genre_choice
    with st.spinner("Finding similar movies..."):
        try:
            recs = tfidf_recommend_titles(query, top_n=top_n, genre_filter=genre_filter)
        except Exception as e:
            recs = None
            st.error(str(e))

    if recs:
        st.markdown(f"### Because you liked *{query}*")
        cols = st.columns(5)
        for i, (title, score) in enumerate(recs):
            poster_url = fetch_poster(title)
            with cols[i % 5]:
                render_movie_card(title, poster_url, f"match {score:.0%}", key_prefix=f"rec-{i}")
    elif recs == []:
        msg = "No recommendations found."
        if genre_filter:
            msg += f" Try removing the '{genre_filter}' genre filter."
        st.warning(msg)
elif go and not query:
    st.warning("Type a movie title first.")
