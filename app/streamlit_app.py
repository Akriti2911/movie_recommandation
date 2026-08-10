"""
Streamlit UI for the TF-IDF movie recommender.

Runs the recommendation logic in-process (no separate FastAPI server needed
for deployment on Streamlit Community Cloud), reusing the same pickled
model artifacts and TMDB helper functions from main.py.
"""

import asyncio

import streamlit as st

from main import (
    TMDB_API_KEY,
    attach_tmdb_card_by_title,
    load_pickles,
    tfidf_recommend_titles,
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

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    query = st.text_input(
        "Type a movie title",
        placeholder="e.g. Toy Story, The Dark Knight, Inception...",
        label_visibility="collapsed",
    )
    top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
    go = st.button("🔍 Find similar movies", type="primary", use_container_width=True)

if go and query:
    with st.spinner("Finding similar movies..."):
        try:
            recs = tfidf_recommend_titles(query, top_n=top_n)
        except Exception as e:
            recs = None
            st.error(str(e))

    if recs:
        st.markdown(f"### Because you liked *{query}*")
        cols = st.columns(5)
        for i, (title, score) in enumerate(recs):
            poster_url = fetch_poster(title)
            with cols[i % 5]:
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
                            <span class="movie-score">match {score:.0%}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    elif recs == []:
        st.warning("No recommendations found.")
elif go and not query:
    st.warning("Type a movie title first.")
