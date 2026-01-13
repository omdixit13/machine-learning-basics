# Simple NLP Text Classification

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample text data
texts = [
    "I love this product",
    "This is a great experience",
    "I hate this service",
    "This is very bad"
]

# Labels: 1 = Positive, 0 = Negative
labels = [1, 1, 0, 0]

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Test prediction
test_text = ["this product is great"]
test_vector = vectorizer.transform(test_text)

prediction = model.predict(test_vector)
print("Prediction (1=Positive, 0=Negative):", prediction[0])
