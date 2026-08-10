"""
Streamlit UI for the TF-IDF movie recommender.

Runs the recommendation logic in-process (no separate FastAPI server needed
for deployment on Streamlit Community Cloud), reusing the same pickled
model artifacts and TMDB helper functions from main.py.
"""

import asyncio
import os

import streamlit as st

from main import (
    TMDB_API_KEY,
    attach_tmdb_card_by_title,
    load_pickles,
    tfidf_recommend_titles,
    tmdb_search_first,
    make_img_url,
)

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource
def init_model():
    load_pickles()
    return True


init_model()


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


st.title("🎬 Movie Recommender")
st.caption("Content-based recommendations using TF-IDF over overview, genres and tagline.")

if not TMDB_API_KEY:
    st.info(
        "No TMDB_API_KEY set — recommendations will still work, just without posters. "
        "Add one in `.env` (local) or Secrets (Streamlit Cloud) to enable posters.",
        icon="ℹ️",
    )

query = st.text_input("Type a movie title", placeholder="e.g. Toy Story")
top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if st.button("Recommend", type="primary") and query:
    with st.spinner("Finding similar movies..."):
        try:
            recs = tfidf_recommend_titles(query, top_n=top_n)
        except Exception as e:
            recs = None
            st.error(str(e))

    if recs:
        st.subheader(f"Because you liked '{query}'")
        cols = st.columns(5)
        for i, (title, score) in enumerate(recs):
            col = cols[i % 5]
            with col:
                poster_url = None
                if TMDB_API_KEY:
                    try:
                        card = run_async(attach_tmdb_card_by_title(title))
                        if card:
                            poster_url = card.poster_url
                    except Exception:
                        poster_url = None

                if poster_url:
                    st.image(poster_url, use_container_width=True)
                st.markdown(f"**{title}**")
                st.caption(f"similarity: {score:.3f}")
    elif recs == []:
        st.warning("No recommendations found.")
