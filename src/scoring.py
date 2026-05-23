from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_similarity(article_embeddings, scope_embedding):
    scores = cosine_similarity(article_embeddings, [scope_embedding]).flatten()
    return scores


def summary_statistics(scores):
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "std": float(np.std(scores))
    }