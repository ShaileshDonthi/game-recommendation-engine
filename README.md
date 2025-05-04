# game-recommendation-engine

## Repository Link
[Steam Game Recommendation Engine](https://github.com/ShaileshDonthi/game-recommendation-engine)

**Aim**:  
To develop a recommendation engine that suggests games based on the game tags and sentiment profiles of Steam reviews.

# Steam Game Recommendation Engine

## Overview
This hybrid recommendation system enhances Steam game discovery by combining game metadata with sentiment analysis of user reviews. Unlike traditional recommenders that rely solely on tags or popularity, this engine introduces **emotion-aware recommendations**.

## Features
- Analyze Steam reviews using SBERT for sentiment representation.
- Recommend games based on a combination of tags (TF-IDF) and review sentiment.
- Hybrid similarity model using cosine distance for both tag and review vectors.
- Plans for explainable recommendations through AI-generated justifications.

## Methodology
1. **Data Collection**: Steam API used to fetch game metadata and user reviews.
2. **Text Embedding**: Reviews are embedded with Sentence-BERT to capture sentiment tone.
3. **Tag Vectorization**: Game tags are transformed using TF-IDF.
4. **Similarity Matching**: Games are compared using cosine similarity across both vectors.
5. **Recommendation**: Outputs similar games based on combined score.

## Technologies Used
- Python
- Sentence-BERT (via `sentence-transformers`)
- Scikit-learn (TF-IDF, cosine similarity)
- Pandas, NumPy, Torch
- Matplotlib/Seaborn (for optional visual analysis)

## Repository Structure
├── data_loader.py # Fetch game metadata and reviews
├── model.py # Core recommendation logic
├── eda_utils.py # Exploratory analysis
├── eval_utils.py # Similarity and evaluation utilities
├── models/
│ ├── game_embeddings.pt # Precomputed SBERT sentiment vectors
│ ├── games_data.csv # Game tags and metadata
│ └── sbert_model/ # Pretrained SBERT model files
├── all_steam_games_with_reviews.csv
├── Setup_Guide.txt # Basic usage instructions
└── README.md


## Running Instructions

Instruction to run this file are in "Setup_Guide.txt"