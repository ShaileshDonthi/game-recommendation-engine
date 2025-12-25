try:
    import pandas as pd
except Exception:
    pd = None


def compute_precision_at_k(recommendations, true_genre, k=5):
    """
    Simple Precision@K based on genre match
    """
    hits = 0
    for _, row in recommendations.head(k).iterrows():
        rec_genres = set(str(row['genres']).split(', '))
        true_genres = set(str(true_genre).split(', '))
        if rec_genres & true_genres:
            hits += 1
    return hits / k

def evaluate_on_sample(df, recommender, k=5, method="bert"):
    """
    Runs evaluation across a few random samples
    """
    # assume df is a pandas-like DataFrame; if pandas not available, helpers should provide data
    if pd is not None:
        games = df.sample(10, random_state=42)
    else:
        # fallback: take first 10 rows from iterable-like df
        games = list(df[:10])
    results = []
    for _, row in games.iterrows():
        try:
            recommendations = recommender.get_similar_games(row["name"], top_n=k, method=method)
            score = compute_precision_at_k(recommendations, row["genres"], k)
            results.append(score)
        except:
            continue
    return sum(results) / len(results) if results else 0.0
