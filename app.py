import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body, .stApp { background-color: #0e1117; color: #ffffff; }

    .movie-card {
        background: #1c1f26;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        transition: transform 0.2s;
    }
    .movie-card:hover { transform: scale(1.03); }

    .badge {
        display: inline-block;
        background: #e50914;
        color: white;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 12px;
        margin: 2px;
    }
    .rating-bar {
        background: #333;
        border-radius: 4px;
        height: 6px;
        margin: 4px 0;
    }
    .rating-fill {
        background: linear-gradient(90deg, #f5c518, #e50914);
        height: 6px;
        border-radius: 4px;
    }
    .section-title {
        font-size: 22px;
        font-weight: bold;
        color: #e50914;
        margin: 20px 0 10px 0;
    }
    div[data-testid="stSelectbox"] label { color: #aaa; }
</style>
""", unsafe_allow_html=True)

# ─── Load Artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    movies            = pd.DataFrame(pickle.load(open('artifacts/movies.pkl', 'rb')))
    tfidf             = pickle.load(open('artifacts/tfidf.pkl', 'rb'))
    content_sim       = pickle.load(open('artifacts/content_similarity.pkl', 'rb'))
    hybrid_sim        = pickle.load(open('artifacts/hybrid_similarity.pkl', 'rb'))
    return movies, tfidf, content_sim, hybrid_sim

movies, tfidf, content_sim, hybrid_sim = load_artifacts()

# ─── TMDB Poster Fetch ───────────────────────────────────────────────────────
TMDB_API_KEY = "16889de71910c8b7c6340daba3741232"   # ← replace with your key

@st.cache_data(ttl=3600)
def fetch_movie_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()

        poster = (
            "https://image.tmdb.org/t/p/w500" + data['poster_path']
            if data.get('poster_path') else
            "https://via.placeholder.com/300x450?text=No+Image"
        )

        return {
            "poster": poster,
            "tagline": data.get("tagline", ""),
            "runtime": data.get("runtime", "N/A"),
            "homepage": data.get("homepage", ""),
        }

    except Exception:
        return {
            "poster": "https://via.placeholder.com/300x450?text=No+Image",
            "tagline": "",
            "runtime": "N/A",
            "homepage": ""
        }

# ─── Core Recommendation Engine ──────────────────────────────────────────────
def recommend(
    movie_title,
    n=10,
    mode="hybrid",           # "hybrid" | "content" | "popular"
    genre_filter=None,
    min_rating=0.0,
    min_year=1900,
):
    sim_matrix = hybrid_sim if mode == "hybrid" else content_sim

    # ── locate the query movie
    match = movies[movies['title'] == movie_title]
    if match.empty:
        return pd.DataFrame()

    idx = match.index[0]
    scores = list(enumerate(sim_matrix[idx]))

    # ── sort, skip self
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [(i, s) for i, s in scores if i != idx]

    # ── build candidate dataframe
    candidate_ids = [i for i, _ in scores[:200]]   # oversample then filter
    sim_scores    = {i: s for i, s in scores[:200]}

    candidates = movies.iloc[candidate_ids].copy()
    candidates['sim_score'] = candidates.index.map(sim_scores)

    # ── filters
    if genre_filter and genre_filter != "All":
        candidates = candidates[
            candidates['genres'].apply(
                lambda g: genre_filter.replace(" ", "") in g
                          if isinstance(g, list) else False
            )
        ]

    candidates = candidates[candidates['vote_average'] >= min_rating]

    if 'release_date' in candidates.columns:
        candidates['year'] = pd.to_datetime(
            candidates['release_date'], errors='coerce'
        ).dt.year.fillna(0).astype(int)
        candidates = candidates[candidates['year'] >= min_year]

    # ── popularity-only mode: re-sort by weighted rating
    if mode == "popular":
        candidates = candidates.sort_values('weighted_rating', ascending=False)

    return candidates.head(n).reset_index(drop=True)


# ─── Render a Row of Movie Cards ─────────────────────────────────────────────
def render_movie_row(df, cols_per_row=5):
    rows = [df.iloc[i:i+cols_per_row] for i in range(0, len(df), cols_per_row)]
    for row in rows:
        cols = st.columns(cols_per_row)
        for col, (_, m) in zip(cols, row.iterrows()):
            details = fetch_movie_details(m['movie_id'])
            rating  = round(m.get('vote_average', 0), 1)
            year    = str(m.get('year', ''))[:4] if 'year' in m else ''
            genres  = m['genres'][:3] if isinstance(m.get('genres'), list) else []

            with col:
                poster = details.get('poster', '')

                # fix markdown format if exists
                if isinstance(poster, str) and poster.startswith('['):
                    try:
                        poster = poster.split('(')[1].replace(')', '')
                    except:
                        poster = ""

                # fallback if invalid
                if not poster or "http" not in poster:
                    poster = "https://via.placeholder.com/300x450?text=No+Image"

                st.image(poster)
                st.markdown(f"**{m['title']}**")

                # genre badges
                badges = "".join(
                    f'<span class="badge">{g}</span>' for g in genres
                )
                st.markdown(badges, unsafe_allow_html=True)

                # rating bar
                bar_pct = int((rating / 10) * 100)
                st.markdown(
                    f"""
                    <div style="font-size:12px;color:#aaa;">
                        ⭐ {rating}/10 &nbsp; {year}
                    </div>
                    <div class="rating-bar">
                        <div class="rating-fill" style="width:{bar_pct}%"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if details.get('runtime') and details['runtime'] != 'N/A':
                    st.caption(f"🕐 {details['runtime']} min")

                if details.get('tagline'):
                    st.caption(f"*{details['tagline'][:60]}*")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio(
    "Recommendation Mode",
    ["Hybrid (Content + Popularity)", "Content Only", "Trending / Popular"],
    index=0
)
mode_map = {
    "Hybrid (Content + Popularity)": "hybrid",
    "Content Only": "content",
    "Trending / Popular": "popular"
}

# Genre filter
all_genres = sorted({
    g.replace(" ", "")
    for genres in movies['genres']
    if isinstance(genres, list)
    for g in genres
})
genre_filter = st.sidebar.selectbox("Genre Filter", ["All"] + all_genres)

# Rating filter
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 5.0, 0.5)

# Year filter
min_year = st.sidebar.slider("Released After", 1950, 2024, 1990)

# Number of results
n_results = st.sidebar.slider("Number of Results", 5, 20, 10)

# ─── Search History ──────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if st.session_state.history:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🕘 Recent Searches**")
    for h in reversed(st.session_state.history[-8:]):
        st.sidebar.caption(f"• {h}")

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#e50914; font-size:38px;'>
    🎬 CineMatch
</h1>
<p style='color:#aaa; margin-top:-10px;'>
    Content-aware movie recommendations powered by TF-IDF + hybrid scoring
</p>
<hr style='border-color:#333;'>
""", unsafe_allow_html=True)

# ─── Trending Section ─────────────────────────────────────────────────────────
with st.expander("🔥 Trending Right Now", expanded=False):
    trending = movies.sort_values('weighted_rating', ascending=False).head(10).copy()
    trending['year'] = pd.to_datetime(
        trending['release_date'], errors='coerce'
    ).dt.year.fillna(0).astype(int)
    render_movie_row(trending, cols_per_row=5)

# ─── Main Search ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔍 Find Movies Like...</div>', unsafe_allow_html=True)

selected_movie = st.selectbox(
    "Type or select a movie title:",
    movies['title'].values,
    label_visibility="collapsed"
)

col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 7])

recommend_clicked = col_btn1.button("🎯 Recommend", use_container_width=True)
save_clicked      = col_btn2.button("💾 Save", use_container_width=True)

if save_clicked:
    if selected_movie not in st.session_state.history:
        st.session_state.history.append(selected_movie)
    st.success(f"Saved *{selected_movie}* to history.")

# ─── Show Selected Movie Info ─────────────────────────────────────────────────
if recommend_clicked or selected_movie:
    query_row = movies[movies['title'] == selected_movie]
    if not query_row.empty:
        q = query_row.iloc[0]
        q_details = fetch_movie_details(q['movie_id'])

        with st.container():
            c1, c2 = st.columns([1, 4])
            with c1:
                poster = q_details.get('poster', '')
                if not poster or "http" not in poster:
                    poster = "https://via.placeholder.com/300x450?text=No+Image"

                st.image(poster, width=160)
            with c2:
                st.markdown(f"### {q['title']}")
                if q_details.get('tagline'):
                    st.caption(f"*{q_details['tagline']}*")

                genres_display = " · ".join(
                    q['genres'][:5] if isinstance(q['genres'], list) else []
                )
                st.markdown(f"**Genres:** {genres_display}")
                st.markdown(
                    f"**Rating:** ⭐ {round(q['vote_average'],1)}/10"
                    f"  |  **Votes:** {int(q['vote_count']):,}"
                    f"  |  **Popularity:** {round(q['popularity'],1)}"
                )

                overview = (
                    " ".join(q['overview'])
                    if isinstance(q['overview'], list) else str(q['overview'])
                )
                st.write(overview[:400] + ("..." if len(overview) > 400 else ""))

# ─── Recommendations ─────────────────────────────────────────────────────────
if recommend_clicked:
    if selected_movie not in st.session_state.history:
        st.session_state.history.append(selected_movie)

    with st.spinner("🔍 Analysing film DNA..."):
        recs = recommend(
            movie_title=selected_movie,
            n=n_results,
            mode=mode_map[mode],
            genre_filter=genre_filter,
            min_rating=min_rating,
            min_year=min_year,
        )

    if recs.empty:
        st.warning("No results matched your filters. Try relaxing the genre or rating settings.")
    else:
        st.markdown(
            f'<div class="section-title">Recommended ({len(recs)} results)</div>',
            unsafe_allow_html=True
        )
        render_movie_row(recs, cols_per_row=5)

# ─── Search by Genre (discovery mode) ────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">🎭 Browse by Genre</div>', unsafe_allow_html=True)

browse_genre = st.selectbox("Pick a genre to explore", [""] + all_genres, key="browse")

if browse_genre:
    genre_movies = movies[
        movies['genres'].apply(
            lambda g: browse_genre in g if isinstance(g, list) else False
        )
    ].sort_values('weighted_rating', ascending=False).head(10).copy()

    genre_movies['year'] = pd.to_datetime(
        genre_movies['release_date'], errors='coerce'
    ).dt.year.fillna(0).astype(int)

    if genre_movies.empty:
        st.info("No movies found for that genre.")
    else:
        render_movie_row(genre_movies, cols_per_row=5)
