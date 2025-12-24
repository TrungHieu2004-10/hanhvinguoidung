# lda_topic.py
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def extract_topics(
    texts: List[str],
    n_topics: int = 5,
    n_top_words: int = 10,
    language: Optional[str] = "english",
    max_df: float = 0.95,
    min_df: int = 2,
) -> List[Dict]:
    """
    Extract topics using CountVectorizer + LDA.

    Args:
        texts: list of documents (strings)
        n_topics: number of topics to extract
        n_top_words: number of top keywords per topic
        language: string for stop_words (e.g., 'english') or None
        max_df: max document frequency for CountVectorizer
        min_df: min document frequency for CountVectorizer (int)

    Returns:
        list of dicts: [{"topic_id": 0, "keywords": "word1, word2, ..."}, ...]
    """
    cleaned_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(cleaned_texts) < 5:
        return []

    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=language)
    X = vectorizer.fit_transform(cleaned_texts)
    if X.shape[1] == 0:
        return []

    n_components = min(n_topics, X.shape[0])
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append({"topic_id": int(topic_idx), "keywords": ", ".join(top_words)})

    return topics
