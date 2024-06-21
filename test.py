from sklearn.calibration import LinearSVC
from sklearn.datasets import make_classification

from linearsvc.main import LinearSVM


X, y = make_classification(n_samples=80, n_features=4, n_informative=2, n_redundant=0, 
                           n_clusters_per_class=1, random_state=42)
y = [1 if label == 1 else -1 for label in y] 

X_train, X_test = X[:50], X[50:]
y_train, y_test = y[:50], y[50:]

sklearn_svm = LinearSVC(C=1.0, random_state=42)
sklearn_svm.fit(X_train, y_train)

my_svm = LinearSVM(C=1.0)
my_svm.fit(y_train, X_train)

# Predict on test data
sklearn_predictions = sklearn_svm.predict(X_test)
our_predictions = my_svm.predict(X_test)

# Print predictions and actual labels
print("Scikit-learn Predictions:", sklearn_predictions)
print("Our LinearSVM Predictions:", our_predictions)
print("Actual labels: ", y_test)

# Calculate accuracy
sklearn_accuracy = sum([pred == actual for pred, actual in zip(sklearn_predictions, y_test)]) / len(y_test)
our_accuracy = sum([pred == actual for pred, actual in zip(our_predictions, y_test)]) / len(y_test)

print(f"Scikit-learn Accuracy: {sklearn_accuracy:.2f}")
print(f"Our LinearSVM Accuracy: {our_accuracy:.2f}")

# Print weight vectors
print("\nScikit-learn Weight Vector:", sklearn_svm.coef_)
print("Our LinearSVM Weight Vector:", my_svm.w)

# Print bias terms
print("\nScikit-learn Bias Term:", sklearn_svm.intercept_)
print("Our LinearSVM Bias Term:", my_svm.b)