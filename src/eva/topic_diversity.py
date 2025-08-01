def compute_diversity(top_words):
    # Flatten và tính tỉ lệ từ duy nhất
    all_words = [word for topic in top_words for word in topic]
    unique_words = set(all_words)
    return len(unique_words) / len(all_words)