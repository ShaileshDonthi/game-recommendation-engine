import streamlit as st
from model import GameRecommender
import pandas as pd
from eda_utils import plot_genre_distribution, generate_wordcloud, plot_sentiment_distribution, plot_pca
from eval_utils import evaluate_on_sample
import os

st.set_page_config(page_title="Steam Game Recommender", layout="wide")

@st.cache_resource
def load_recommender():
    if os.path.exists("models/games_data.csv"):
        return GameRecommender.load_model()
    else:
        st.error("Model not found. Please run model.py to train and save it.")
        return None

recommender = load_recommender()

# Navigation tabs
tabs = st.tabs(["🎮 Recommendations", "📊 EDA", "🧪 Evaluation"])

with tabs[0]:
    st.title("🎮 Game Recommendations")
    if recommender and recommender.df is not None:
        game_names = sorted(recommender.df["name"].unique())
        selected_game = st.selectbox("Select a game:", game_names)
        method = st.radio("Model", ["BERT", "TF-IDF", "Hybrid"], horizontal=True)
        top_n = st.slider("Number of recommendations", 3, 10, 5)
        apply_sentiment = st.checkbox("Only show positive sentiment games")
        sentiment_thresh = 0.2 if apply_sentiment else None

        if st.button("Get Recommendations"):
            try:
                recs = recommender.get_similar_games(selected_game, top_n=top_n, method=method.lower(), sentiment_threshold=sentiment_thresh)
                st.dataframe(recs, use_container_width=True)
            except Exception as e:
                st.error(str(e))

with tabs[1]:
    st.title("📊 EDA & Visual Insights")
    if recommender and recommender.df is not None:
        plot_genre_distribution(recommender.df)
        generate_wordcloud(recommender.df)
        plot_sentiment_distribution(recommender.df)
        if recommender.embeddings is not None:
            st.subheader("PCA Cluster View of Embeddings")
            plot_pca(recommender.embeddings)

with tabs[2]:
    st.title("🧪 Evaluate Model Performance")
    if recommender and recommender.df is not None:
        eval_method = st.selectbox("Select model to evaluate", ["bert", "tfidf", "hybrid"])
        if st.button("Run Precision@K on Sample (k=5)"):
            score = evaluate_on_sample(recommender.df, recommender, k=5, method=eval_method)
            st.success(f"Average Precision@5: {score:.2f}")
