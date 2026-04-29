# Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset (movies.csv in same folder)
df = pd.read_csv("movies.csv")

# Keep required columns
df = df[['movie_title', 'genres']]

# Data Cleaning
df['movie_title'] = df['movie_title'].str.strip()
df['genres'] = df['genres'].fillna('')

# Remove duplicate movies
df = df.drop_duplicates(subset='movie_title')

# Reset index
df = df.reset_index(drop=True)

# Convert text data (genres) into numerical form
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Recommendation Function
def recommend(movie_name, top_n=5):
    movie_name = movie_name.strip()
    
    # Check if movie exists
    if movie_name not in df['movie_title'].values:
        print("Movie not found! Try another name.")
        return
    
    # Get index of movie
    idx = df[df['movie_title'] == movie_name].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort based on similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Print recommendations
    print(f"\nTop {top_n} recommendations for '{movie_name}':\n")
    
    count = 0
    for i in sim_scores[1:]:
        print(df['movie_title'].iloc[i[0]])
        count += 1
        if count >= top_n:
            break

# Example Usage
recommend("Avatar")