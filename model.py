from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model():
    """Train a Logistic Regression classifier on the built-in Iris dataset."""
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, X_test, y_test, accuracy


if __name__ == "__main__":
    model, X_test, y_test, accuracy = train_model()
    print("Model: Logistic Regression")
    print("Dataset: Iris")
    print(f"Accuracy: {accuracy:.4f}")
    print("Model training and prediction completed.")
