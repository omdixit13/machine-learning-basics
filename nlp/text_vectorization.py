# Basic NLP: Text Vectorization

from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "AI will change the world",
    "Machine learning is part of AI",
    "Python is used in machine learning"
]

vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(texts)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Vectors:\n", vectors.toarray())
