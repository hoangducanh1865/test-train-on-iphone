import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer

def compute_coherence(topics, vectorizer: CountVectorizer):
    """
    Tính coherence score sơ bộ bằng PMI giữa các từ trong topic.
    Không cần raw text, chỉ dùng ma trận BOW và co-occurrence.
    """
    # Lấy corpus dưới dạng sparse matrix (docs x vocab)
    X = vectorizer.transform(vectorizer.inverse_transform(np.eye(vectorizer.vocabulary_.__len__())))
    X = X.toarray()

    score = 0.0
    count = 0

    for topic in topics:
        for w1, w2 in itertools.combinations(topic, 2):
            if w1 not in vectorizer.vocabulary_ or w2 not in vectorizer.vocabulary_:
                continue
            i, j = vectorizer.vocabulary_[w1], vectorizer.vocabulary_[w2]
            p_i = X[:, i].sum() + 1
            p_j = X[:, j].sum() + 1
            p_ij = (X[:, i] * X[:, j]).sum() + 1
            score += np.log(p_ij / (p_i * p_j))
            count += 1

    return score / count if count else 0.0