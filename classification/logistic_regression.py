from sklearn.linear_model import LogisticRegression
import numpy as np

# Age vs purchase decision
X = np.array([[18], [22], [25], [30], [35]])
y = np.array([0, 0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

result = model.predict([[28]])
print("Prediction result:", result[0])
