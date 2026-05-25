""" Find similar questions in a DataFrame. This module defines a function to identify similar questions in a training dataset
using TF-IDF vectorization and cosine similarity. It prevents data leakage by ensuring validation questions are not included in
the training set and assigns group IDs to clusters of similar questions for further analysis. """

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import gc


def find_similar_questions(train_df):
    cols_to_check = ['prompt', 'A', 'B', 'C', 'D', 'E', 'answer']
    train_df = train_df[~train_df.duplicated(subset=cols_to_check, keep='first')].copy().reset_index(drop=True)

    full_text = (
        train_df['prompt'].fillna("") + " " + train_df['A'].fillna("") + " " +
        train_df['B'].fillna("") + " " + train_df['C'].fillna("") + " " +
        train_df['D'].fillna("") + " " + train_df['E'].fillna("")
    )

    # TF-IDF vectorization and cosine similarity calculation to find similar questions
    vectorizer   = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(full_text)
    sim_matrix   = cosine_similarity(tfidf_matrix)

    # Build a graph where nodes represent questions and edges represent high similarity between questions
    G = nx.Graph()
    G.add_nodes_from(range(len(train_df)))
    high_sim_indices = np.where(np.triu(sim_matrix, k=1) > 0.85)
    for i, j in zip(high_sim_indices[0], high_sim_indices[1]):
        G.add_edge(i, j)

    # Assign group IDs to connected components in the graph, which represent clusters of similar questions
    train_df['group_id'] = -1
    for group_id, component in enumerate(nx.connected_components(G)):
        for node in component:
            train_df.loc[node, 'group_id'] = group_id

    # Clean up memory
    del full_text, vectorizer, tfidf_matrix, sim_matrix, G
    gc.collect()

    return train_df
