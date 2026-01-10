from sklearn.model_selection import train_test_split

# Sample dataset
data = [1, 2, 3, 4, 5, 6, 7, 8]
labels = [2, 4, 6, 8, 10, 12, 14, 16]

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.25,
    random_state=42
)

print("Training data:", X_train)
print("Testing data:", X_test)
