import numpy as np
import pytest
from src.knn import KNNClassifier, euclidean_distance, manhattan_distance


def test_knn_k1_nearest_neighbor():
    """k=1: должен вернуть метку ближайшего соседа."""
    X_train = np.array([[0, 0], [5, 5], [10, 10]], dtype=float)
    y_train = np.array([0, 1, 2])

    clf = KNNClassifier(k=1)
    clf.fit(X_train, y_train)

    X_test = np.array([[6, 6]], dtype=float)
    y_pred = clf.predict(X_test)

    assert y_pred[0] == 1


def test_knn_k3_majority_vote():
    """k=3, uniform: класс определяется большинством голосов."""
    X_train = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11]], dtype=float)
    y_train = np.array([0, 1, 1, 2, 2])

    clf = KNNClassifier(k=3, weights='uniform')
    clf.fit(X_train, y_train)

    # 3 ближайших к [1.5, 1.5]: [1,1](1), [2,2](1), [0,0](0) → класс 1
    X_test = np.array([[1.5, 1.5]])
    y_pred = clf.predict(X_test)

    assert y_pred[0] == 1


def test_knn_manhattan_metric():
    """Проверка работы с манхэттенской метрикой."""
    X_train = np.array([[0, 0], [3, 0], [0, 3]], dtype=float)
    y_train = np.array([0, 1, 2])

    clf = KNNClassifier(k=1, metric=manhattan_distance)
    clf.fit(X_train, y_train)

    # manhattan: до [0,0]=2, до [3,0]=1, до [0,3]=5 → класс 1
    X_test = np.array([[2, 0]], dtype=float)
    y_pred = clf.predict(X_test)

    assert y_pred[0] == 1


def test_knn_custom_metric():
    """Проверка работы с пользовательской функцией расстояния."""
    def chebyshev_distance(x, y):
        return float(np.max(np.abs(x - y)))

    X_train = np.array([[0, 0], [3, 1], [1, 3]], dtype=float)
    y_train = np.array([0, 1, 2])

    clf = KNNClassifier(k=1, metric=chebyshev_distance)
    clf.fit(X_train, y_train)

    # chebyshev: до [0,0]=2, до [3,1]=1, до [1,3]=1 → один из {1, 2}
    # до [3,1]: max(|2-3|, |0-1|)=1; до [1,3]: max(|2-1|, |0-3|)=3
    # → ближайший [3,1], класс 1
    X_test = np.array([[2, 0]], dtype=float)
    y_pred = clf.predict(X_test)

    assert y_pred[0] == 1


def test_knn_uniform_vs_distance_weights():
    """uniform и distance дают разные результаты в пограничном случае."""
    X_train = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])
    y_train = np.array([1, 0, 0])

    clf_uniform = KNNClassifier(k=3, weights='uniform')
    clf_uniform.fit(X_train, y_train)

    clf_distance = KNNClassifier(k=3, weights='distance')
    clf_distance.fit(X_train, y_train)

    X_test = np.array([[0.01, 0.0]])

    # uniform: класс 0 (2 голоса) vs класс 1 (1 голос) → 0
    assert clf_uniform.predict(X_test)[0] == 0
    # distance: вес класса 1 ≈ 100 >> вес класса 0 ≈ 11.2 → 1
    assert clf_distance.predict(X_test)[0] == 1


def test_knn_multiple_predictions():
    """predict возвращает массив правильной длины с корректными метками."""
    X_train = np.array([[0, 0], [10, 10]], dtype=float)
    y_train = np.array([0, 1])

    clf = KNNClassifier(k=1)
    clf.fit(X_train, y_train)

    X_test = np.array([[1, 1], [9, 9], [0, 0]], dtype=float)
    y_pred = clf.predict(X_test)

    assert len(y_pred) == 3
    assert y_pred[0] == 0
    assert y_pred[1] == 1
    assert y_pred[2] == 0


def test_knn_k5_multiclass():
    """k=5, три класса: проверка корректного голосования."""
    X_train = np.array([
        [10, 10], [10.1, 10], [10, 10.1],   # класс 0 (3 объекта)
        [5, 5], [5.1, 5],                    # класс 1 (2 объекта)
        [0, 0], [0.1, 0.1], [0.2, 0],       # класс 2 (3 объекта)
    ], dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 2, 2, 2])

    clf = KNNClassifier(k=5)
    clf.fit(X_train, y_train)

    # Ближайшие 5 к [0.05, 0.05]: три класса-2 + два класса-1 → класс 2
    y_pred = clf.predict(np.array([[0.05, 0.05]]))
    assert y_pred[0] == 2
