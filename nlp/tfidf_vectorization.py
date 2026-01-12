# NLP: TF-IDF Vectorization

from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "AI is the future",
    "AI and machine learning are related",
    "Python is popular for AI"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())
