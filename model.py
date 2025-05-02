import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import joblib

class GameRecommender:
    def __init__(self, data_path=None, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.df = None
        self.embeddings = None
        self.faiss_index = None
        self.tfidf_matrix = None
        self.vectorizer = None
        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path):
        self.df = pd.read_csv(data_path)
        for col in ['reviews', 'genres', 'short_description']:
            self.df[col] = self.df[col].fillna('')
        self.df["combined_text"] = (
            self.df["reviews"] + " " +
            self.df["genres"] + " " +
            self.df["short_description"]
        ).str.replace('\x00', '', regex=False).astype(str)

    def generate_embeddings(self, save_path=None):
        print("Generating BERT embeddings...")
        self.embeddings = self.model.encode(
            self.df["combined_text"].tolist(),
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=True
        ).cpu().detach().numpy()

        if save_path:
            torch.save(self.embeddings, save_path)

        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(self.embeddings)

    def generate_tfidf_matrix(self):
        print("Generating TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])

    def get_top_keywords(self, text, top_n=3):
        if not self.vectorizer:
            raise ValueError("TF-IDF vectorizer not initialized.")
        vec = self.vectorizer.transform([text])
        scores = vec.toarray().flatten()
        indices = np.argsort(scores)[-top_n:][::-1]
        return [self.vectorizer.get_feature_names_out()[i] for i in indices]

    def get_similar_games(self, game_name, top_n=5, method="bert", sentiment_threshold=None):
        matches = self.df[self.df["name"].str.lower() == game_name.lower()]
        if matches.empty:
            matches = self.df[self.df["name"].str.lower().str.contains(game_name.lower())]
        if matches.empty:
            raise ValueError(f"Game '{game_name}' not found.")
        game_idx = matches.index[0]

        if method in ["bert", "hybrid"] and self.embeddings is None:
            self.generate_embeddings()
        if method in ["tfidf", "hybrid"] and self.tfidf_matrix is None:
            self.generate_tfidf_matrix()

        sim_scores = {}
        if method in ["bert", "hybrid"]:
            query = self.embeddings[game_idx].reshape(1, -1)
            dists, indices = self.faiss_index.search(query, top_n + 10)
            indices, dists = indices[0][1:], dists[0][1:]
            bert_scores = {i: 1 / (1 + d) for i, d in zip(indices, dists)}
            sim_scores["bert"] = bert_scores

        if method in ["tfidf", "hybrid"]:
            tfidf_sims = cosine_similarity(self.tfidf_matrix[game_idx], self.tfidf_matrix).flatten()
            tfidf_indices = np.argsort(tfidf_sims)[-top_n - 10:-1][::-1]
            tfidf_scores = {i: tfidf_sims[i] for i in tfidf_indices}
            sim_scores["tfidf"] = tfidf_scores

        combined_scores = {}
        all_indices = set()
        for scores in sim_scores.values():
            all_indices |= scores.keys()

        for idx in all_indices:
            score = 0
            if method == "bert":
                score = sim_scores["bert"].get(idx, 0)
            elif method == "tfidf":
                score = sim_scores["tfidf"].get(idx, 0)
            elif method == "hybrid":
                b = sim_scores["bert"].get(idx, 0)
                t = sim_scores["tfidf"].get(idx, 0)
                score = (b + t) / 2
            combined_scores[idx] = score

        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in sorted_indices if idx != game_idx]

        result_df = self.df.iloc[top_indices[:top_n]].copy()
        result_df["similarity_score"] = [combined_scores[i] for i in top_indices[:top_n]]

        if method in ["tfidf", "hybrid"]:
            result_df["top_keywords"] = result_df["combined_text"].apply(
                lambda x: ", ".join(self.get_top_keywords(x))
            )

        if sentiment_threshold is not None:
            if "sentiment_score" not in result_df.columns:
                raise ValueError("Sentiment filtering requested, but 'sentiment_score' column not found.")
            result_df = result_df[result_df["sentiment_score"] >= sentiment_threshold]

        required_cols = ["name", "genres", "release_date", "short_description", "similarity_score"]
        for col in required_cols:
            if col not in result_df.columns:
                result_df[col] = "N/A"

        return result_df[required_cols + (["top_keywords"] if "top_keywords" in result_df else [])]

    def save_model(self, dir_path="models"):
        os.makedirs(dir_path, exist_ok=True)
        self.df.to_csv(os.path.join(dir_path, "games_data.csv"), index=False)
        if self.embeddings is not None:
            torch.save(self.embeddings, os.path.join(dir_path, "game_embeddings.pt"))
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, os.path.join(dir_path, "tfidf_vectorizer.joblib"))

    @classmethod
    def load_model(cls, dir_path="models"):
        recommender = cls()
        data_path = os.path.join(dir_path, "games_data.csv")
        if os.path.exists(data_path):
            recommender.load_data(data_path)
        emb_path = os.path.join(dir_path, "game_embeddings.pt")
        if os.path.exists(emb_path):
            recommender.embeddings = torch.load(emb_path)
            dim = recommender.embeddings.shape[1]
            recommender.faiss_index = faiss.IndexFlatL2(dim)
            recommender.faiss_index.add(recommender.embeddings)
        tfidf_path = os.path.join(dir_path, "tfidf_vectorizer.joblib")
        if os.path.exists(tfidf_path):
            recommender.vectorizer = joblib.load(tfidf_path)
        return recommender



if __name__ == "__main__":
    csv_path = "all_steam_games_with_reviews.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}. Please run data_loader.py first.")

    recommender = GameRecommender(csv_path)
    recommender.generate_embeddings("game_embeddings.pt")
    recommender.generate_tfidf_matrix()
    recommender.save_model()
    print("Model training and saving complete.")
