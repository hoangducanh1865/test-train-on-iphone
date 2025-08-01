import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer

def compute_coherence(topics, vectorizer: CountVectorizer):
    """
    Approximate topic coherence by computing PMI over vectorizer's vocabulary.
    """
    vocab = vectorizer.get_feature_names_out()
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    X = vectorizer.transform(vectorizer._args['input_data']).toarray()  # raw corpus texts

    score = 0.0
    count = 0

    for topic in topics:
        for w1, w2 in itertools.combinations(topic, 2):
            if w1 not in vocab_index or w2 not in vocab_index:
                continue
            i, j = vocab_index[w1], vocab_index[w2]
            p_i = X[:, i].sum() + 1
            p_j = X[:, j].sum() + 1
            p_ij = (X[:, i] * X[:, j]).sum() + 1
            score += np.log(p_ij / (p_i * p_j))
            count += 1

    return score / count if count else 0.0