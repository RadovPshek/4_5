import numpy as np
import pytest
from src.svm import LinearSVM


def test_svm_linearly_separable():
    """SVM должен идеально разделить линейно разделимые данные."""
    np.random.seed(42)
    X_pos = np.random.randn(50, 2) + np.array([3, 3])
    X_neg = np.random.randn(50, 2) + np.array([-3, -3])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [-1] * 50)

    svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X, y)

    y_pred = svm.predict(X)
    accuracy = np.mean(y_pred == y)

    assert accuracy >= 0.95


def test_svm_weights_after_fit():
    """После fit веса w и смещение b должны быть инициализированы."""
    X = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    y = np.array([1, 1, -1, -1])

    svm = LinearSVM(n_iters=100)
    svm.fit(X, y)

    assert svm.w is not None
    assert svm.b is not None
    assert svm.w.shape == (2,)


def test_svm_predict_shape():
    """predict возвращает массив нужной длины."""
    X_train = np.array([[1, 2], [3, 4], [-1, -2], [-3, -4]], dtype=float)
    y_train = np.array([1, 1, -1, -1])

    svm = LinearSVM(n_iters=200)
    svm.fit(X_train, y_train)

    X_test = np.array([[0, 0], [5, 5], [-5, -5]], dtype=float)
    y_pred = svm.predict(X_test)

    assert y_pred.shape == (3,)


def test_svm_predict_labels():
    """predict возвращает только метки из {-1, +1}."""
    np.random.seed(0)
    X = np.random.randn(40, 3)
    y = np.array([1 if x[0] > 0 else -1 for x in X])

    svm = LinearSVM(learning_rate=0.01, lambda_param=0.01, n_iters=500)
    svm.fit(X, y)

    y_pred = svm.predict(X)
    assert set(np.unique(y_pred)).issubset({-1, 1})


def test_svm_decision_function_shape():
    """decision_function возвращает массив нужной длины."""
    X_train = np.array([[1, 0], [-1, 0]], dtype=float)
    y_train = np.array([1, -1])

    svm = LinearSVM(n_iters=100)
    svm.fit(X_train, y_train)

    scores = svm.decision_function(X_train)
    assert scores.shape == (2,)


def test_svm_decision_function_consistency():
    """Предсказания predict совпадают со знаком decision_function."""
    np.random.seed(42)
    X_pos = np.random.randn(30, 2) + np.array([3, 3])
    X_neg = np.random.randn(30, 2) + np.array([-3, -3])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 30 + [-1] * 30)

    svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=500)
    svm.fit(X, y)

    scores = svm.decision_function(X)
    y_pred = svm.predict(X)

    expected = np.where(scores >= 0, 1, -1)
    assert np.array_equal(y_pred, expected)
