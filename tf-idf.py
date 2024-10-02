import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan

df = pd.read_excel('interview_lemmatized.xlsx', engine="openpyxl", index_col = 0)
documents = df['paragraphs']


tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
feature_names = tfidf_vectorizer.get_feature_names_out()

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(tfidf_matrix)

top_n_words = 10
topics = {}

for cluster_id in np.unique(cluster_labels):
    if cluster_id == -1:
        continue  
    
    mask = cluster_labels == cluster_id
    cluster_mean = np.array(tfidf_matrix[mask].mean(axis=0)).flatten()
    sorted_indices = np.argsort(cluster_mean)[::-1]
    top_features = [(cluster_mean[i], feature_names[i]) for i in sorted_indices[:top_n_words]]
    topic_description = " + ".join([f"{weight:.4f}*{word}" for weight, word in top_features])
    topics[f"Topic{cluster_id}"] = topic_description

for topic, description in topics.items():
    print(f"{topic}: {description}")