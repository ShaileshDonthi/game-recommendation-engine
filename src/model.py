# src/model.py
"""
This module contains functions to build the recommendation model
using TF-IDF vectorization and cosine similarity.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_tfidf_matrix(df, text_column):
    """
    Create a TF-IDF matrix from the text data in the specified DataFrame column.
    
    :param df: DataFrame containing game data.
    :param text_column: str, the name of the column with text to vectorize.
    :return: tuple (tfidf_matrix, vectorizer)
    """
    # Initialize the vectorizer with English stop words.
    # We also fill missing values with an empty string to avoid errors.
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[text_column].fillna(''))
    return tfidf_matrix, tfidf

def get_similar_games(tfidf_matrix, game_index, top_n=5):
    """
    Compute cosine similarity to find the most similar games.
    
    :param tfidf_matrix: Sparse matrix of TF-IDF features.
    :param game_index: int, index of the game for which recommendations are required.
    :param top_n: int, number of recommendations to return.
    :return: list of indices of the top similar games.
    """
    # Compute cosine similarity between the target game and all games
    cosine_sim = cosine_similarity(tfidf_matrix[game_index], tfidf_matrix)
    # Exclude the target game itself and get the indices of the top similar games
    similar_indices = cosine_sim[0].argsort()[-top_n-1:-1][::-1]
    return similar_indices
