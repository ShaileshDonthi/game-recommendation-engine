import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import streamlit as st

def plot_genre_distribution(df):
    genre_series = df['genres'].dropna().str.split(', ').explode()
    top_genres = genre_series.value_counts().head(15)
    fig, ax = plt.subplots()
    sns.barplot(y=top_genres.index, x=top_genres.values, ax=ax)
    ax.set_title("Top 15 Genres")
    ax.set_xlabel("Count")
    ax.set_ylabel("Genre")
    st.pyplot(fig)

def generate_wordcloud(df):
    text = " ".join(df['short_description'].dropna().tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def plot_sentiment_distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(df['sentiment_score'], kde=True, bins=30, ax=ax)
    ax.set_title("Sentiment Score Distribution")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def plot_pca(embeddings, labels=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c='blue', alpha=0.6)
    ax.set_title("PCA of Game Embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)
