# Simple Recommendation System using Content-Based Filtering

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset: movies with genres
data = {
    "Movie": [
        "The Dark Knight", "Inception", "Interstellar", 
        "Avengers: Endgame", "Iron Man", "The Matrix", 
        "Titanic", "The Notebook", "La La Land", "Joker"
    ],
    "Genre": [
        "Action Crime Drama", "Action Sci-Fi Thriller", "Adventure Drama Sci-Fi",
        "Action Adventure Sci-Fi", "Action Adventure Sci-Fi",
        "Action Sci-Fi", "Drama Romance", "Drama Romance", 
        "Comedy Drama Romance", "Crime Drama Thriller"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert genres into numerical vectors (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
genre_matrix = vectorizer.fit_transform(df["Genre"])

# Compute similarity between all movies
similarity = cosine_similarity(genre_matrix)

# Function to recommend movies
def recommend(movie_name, n=3):
    if movie_name not in df["Movie"].values:
        return f"Movie '{movie_name}' not found in database."
    
    # Get index of given movie
    idx = df[df["Movie"] == movie_name].index[0]
    
    # Get similarity scores for this movie
    scores = list(enumerate(similarity[idx]))
    
    # Sort movies based on similarity (ignore itself)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    # Get recommended movies
    recommended = [df.iloc[i[0]]["Movie"] for i in scores]
    return recommended

# Example usage
print("Recommendations for 'Inception':")
print(recommend("Inception"))

print("\nRecommendations for 'Titanic':")
print(recommend("Titanic"))


