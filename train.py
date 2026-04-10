import pandas as pd
import ast
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# ─── Load & Merge ────────────────────────────────────────────────────────────
movies = pd.read_csv(r'C:\projects\movie recommender project\data\tmdb_5000_movies.csv')
credits  = pd.read_csv(r'C:\projects\movie recommender project\data\tmdb_5000_credits.csv')
movies   = movies.merge(credits, on='title')

# ─── Select & Clean Columns ──────────────────────────────────────────────────
movies = movies[[
    'movie_id', 'title', 'overview', 'genres', 'keywords',
    'cast', 'crew', 'vote_average', 'vote_count', 'popularity', 'release_date'
]]
movies.dropna(inplace=True)
movies.drop_duplicates(subset='title', inplace=True)

# ─── Parse JSON-like Columns ─────────────────────────────────────────────────
def parse_list(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def parse_cast(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:5]]  # top 5 cast

def parse_director(obj):
    return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']

movies['genres']   = movies['genres'].apply(parse_list)
movies['keywords'] = movies['keywords'].apply(parse_list)
movies['cast']     = movies['cast'].apply(parse_cast)
movies['crew']     = movies['crew'].apply(parse_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# ─── Remove spaces from multi-word names so they're treated as single tokens ─
def collapse(lst):
    return [i.replace(" ", "") for i in lst]

movies['genres']   = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast']     = movies['cast'].apply(collapse)
movies['crew']     = movies['crew'].apply(collapse)

# ─── Weighted Tags (genres & director get boosted repetitions) ────────────────
# Repeating a token is a simple but effective way to up-weight it in TF-IDF
movies['tags'] = (
    movies['overview']
    + movies['genres'] * 3          # genre gets 3× weight
    + movies['keywords'] * 2
    + movies['cast']
    + movies['crew'] * 2            # director gets 2× weight
)
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# ─── Stemming ────────────────────────────────────────────────────────────────
ps = PorterStemmer()

def stem(text):
    return " ".join(ps.stem(w) for w in text.split())

movies['tags'] = movies['tags'].apply(stem)
movies['tags'] = movies['tags'].str.lower()

# ─── TF-IDF Vectorisation ────────────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(movies['tags']).toarray()

# ─── Content-Based Cosine Similarity ─────────────────────────────────────────
content_similarity = cosine_similarity(vectors)   # shape: (n, n)

# ─── Popularity Score (normalised 0–1) ───────────────────────────────────────
# Used later to break ties and for trending section
movies['norm_popularity'] = (
    movies['popularity'] / movies['popularity'].max()
)
movies['norm_votes'] = (
    movies['vote_count'] / movies['vote_count'].max()
)

# Weighted rating (Bayesian average — same formula IMDB uses)
C = movies['vote_average'].mean()   # mean rating across all movies
m = movies['vote_count'].quantile(0.70)  # minimum votes threshold

def weighted_rating(row, m=m, C=C):
    v = row['vote_count']
    R = row['vote_average']
    return (v / (v + m)) * R + (m / (v + m)) * C

movies['weighted_rating'] = movies.apply(weighted_rating, axis=1)
movies['norm_rating'] = movies['weighted_rating'] / movies['weighted_rating'].max()

# ─── Hybrid Score Matrix ──────────────────────────────────────────────────────
# Blend content similarity with a small popularity nudge
# score = 0.80 * content_sim + 0.10 * popularity + 0.10 * rating
pop_matrix    = np.outer(movies['norm_popularity'].values, np.ones(len(movies)))
rating_matrix = np.outer(movies['norm_rating'].values, np.ones(len(movies)))

hybrid_similarity = (
    0.80 * content_similarity
    + 0.10 * pop_matrix
    + 0.10 * rating_matrix
)

# ─── Pickle Artifacts ────────────────────────────────────────────────────────
movies.reset_index(drop=True, inplace=True)

pickle.dump(movies,            open('artifacts/movies.pkl',            'wb'))
pickle.dump(tfidf,             open('artifacts/tfidf.pkl',             'wb'))
pickle.dump(content_similarity,open('artifacts/content_similarity.pkl','wb'))
pickle.dump(hybrid_similarity, open('artifacts/hybrid_similarity.pkl', 'wb'))

print("✅ Training complete. Artifacts saved to /artifacts")
print(f"   Movies in index : {len(movies)}")
print(f"   Vocabulary size : {len(tfidf.vocabulary_)}")
print(f"   Similarity shape: {hybrid_similarity.shape}")
