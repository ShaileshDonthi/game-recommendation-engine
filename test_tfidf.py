from model import GameRecommender

r = GameRecommender.load_model()
print('Loaded df:', r.df is not None)
print('Vectorizer present:', r.vectorizer is not None)

sample = r.df['name'].iloc[0]
print('Sample game:', sample)
try:
    res = r.get_similar_games(sample, top_n=3, method='TF-IDF')
    print(res[['name','similarity_score']])
except Exception as e:
    print('Error:', e)
