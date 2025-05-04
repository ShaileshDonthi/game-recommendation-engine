# game-recommendation-engine

## Repository Link
[Sentiment-Driven Game Recommendation Engine](https://github.com/ShaileshDonthi/game-recommendation-engine)

**Aim**:  
To develop a recommendation engine that suggests games based on gameplay tags and semantic profiles of Steam reviews, using embedded emotional and contextual meaning rather than explicit sentiment classification.

# Sentiment-Driven Game Recommendation Engine

## Overview
This hybrid recommendation system enhances Steam game discovery by combining game metadata with review embedding analysis using Sentence-BERT (SBERT). Unlike traditional recommenders that rely solely on tags or popularity, this engine captures emotional tone and thematic similarity from player reviews.

## Features
- Use SBERT to embed review text, capturing semantic and emotional meaning.
- Recommend games based on a combination of TF-IDF tag similarity and SBERT review embeddings.
- Hybrid similarity model using cosine distance for both tag and review vectors.
- Plans for explainable recommendations through AI-generated justifications.

## Methodology
1. **Data Collection**: Steam API used to fetch game metadata and user reviews.
2. **Text Embedding**: Reviews are embedded with Sentence-BERT to capture semantic tone.
3. **Tag Vectorization**: Game tags are transformed using TF-IDF.
4. **Similarity Matching**: Games are compared using cosine similarity across both vectors.
5. **Recommendation**: Outputs similar games based on combined score.

## Technologies Used
- Python
- Sentence-BERT (via `sentence-transformers`)
- Scikit-learn (TF-IDF, cosine similarity)
- Pandas, NumPy, Torch
- Streamlit (for interactive user interface)
- Matplotlib/Seaborn (for optional visual analysis)


## Repository Structure


## Running Instructions

Instruction to run this file are in "Setup_Guide.txt"