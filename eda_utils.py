try:
    import numpy as np
except Exception:
    np = None

# Optional plotting / data deps — import lazily or provide safe fallbacks
try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except Exception:
    plt = None
    _MPL_AVAILABLE = False

try:
    import seaborn as sns
    _SNS_AVAILABLE = True
except Exception:
    sns = None
    _SNS_AVAILABLE = False

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from sklearn.decomposition import PCA
    _PCA_AVAILABLE = True
except Exception:
    PCA = None
    _PCA_AVAILABLE = False

try:
    import streamlit as st
except Exception:
    # minimal shim for streamlit.pyplot when streamlit isn't installed
    class _StreamlitShim:
        def pyplot(self, fig=None):
            if fig is None:
                if _MPL_AVAILABLE:
                    fig = plt.gcf()
                else:
                    return
            try:
                if _MPL_AVAILABLE:
                    plt.show()
            except Exception:
                try:
                    fig.savefig('plot.png')
                except Exception:
                    pass

    st = _StreamlitShim()

def plot_genre_distribution(df):
    if not _MPL_AVAILABLE or not _SNS_AVAILABLE:
        print("matplotlib or seaborn not available — skipping genre distribution plot")
        return
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
    if WordCloud is None:
        print("wordcloud package not available; skipping wordcloud generation")
        return
    if not _MPL_AVAILABLE:
        print("matplotlib not available; skipping wordcloud rendering")
        return
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    # Use the PIL image returned by WordCloud.to_image() to avoid calling
    # numpy.asarray with unsupported kwargs in some numpy versions.
    wc_image = wordcloud.to_image()
    ax.imshow(wc_image, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def plot_sentiment_distribution(df):
    if not _MPL_AVAILABLE or not _SNS_AVAILABLE:
        print("matplotlib or seaborn not available — skipping sentiment distribution plot")
        return
    fig, ax = plt.subplots()
    sns.histplot(df['sentiment_score'], kde=True, bins=30, ax=ax)
    ax.set_title("Sentiment Score Distribution")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def plot_pca(embeddings, labels=None):
    if not _MPL_AVAILABLE or not _PCA_AVAILABLE:
        print("matplotlib or sklearn PCA not available — skipping PCA plot")
        return
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c='blue', alpha=0.6)
    ax.set_title("PCA of Game Embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)
