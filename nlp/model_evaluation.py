# NLP Model Evaluation Example

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

texts = [
    "I love this product",
    "Amazing experience",
    "Very bad service",
    "I hate this item"
]

labels = [1, 1, 0, 0]  # 1 = Positive, 0 = Negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

predictions = model.predict(X)

print("Accuracy:", accuracy_score(labels, predictions))
print("Report:\n", classification_report(labels, predictions))
