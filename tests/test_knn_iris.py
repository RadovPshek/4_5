import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.knn import KNNClassifier, euclidean_distance, manhattan_distance


def test_knn_euclidean_on_iris():
    """kNN (k=5, euclidean) на Iris: accuracy >= 0.8."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    clf = KNNClassifier(k=5, metric=euclidean_distance)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc >= 0.8


def test_knn_manhattan_on_iris():
    """kNN (k=5, manhattan) на Iris: accuracy >= 0.8."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    clf = KNNClassifier(k=5, metric=manhattan_distance)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc >= 0.8


def test_knn_distance_weights_on_iris():
    """kNN (k=5, distance weights) на Iris: accuracy >= 0.8."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    clf = KNNClassifier(k=5, metric=euclidean_distance, weights='distance')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    assert acc >= 0.8
