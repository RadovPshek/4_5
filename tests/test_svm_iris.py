import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.svm import LinearSVM


def test_svm_binary_iris():
    """LinearSVM на двух классах Iris (0 vs 1): accuracy >= 0.85."""
    data = load_iris()

    mask = data.target != 2
    X = data.data[mask]
    y = data.target[mask]
    y = np.where(y == 0, -1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc >= 0.85


def test_svm_binary_iris_classes_1_vs_2():
    """LinearSVM на двух классах Iris (1 vs 2): accuracy >= 0.75."""
    data = load_iris()

    mask = data.target != 0
    X = data.data[mask]
    y = data.target[mask]
    y = np.where(y == 1, -1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc >= 0.75
