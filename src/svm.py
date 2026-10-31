import numpy as np


# Класс линейного метода опорных векторов (SVM)
class LinearSVM:
    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000):
        self.learning_rate = learning_rate  # Скорость обучения
        self.lambda_param = lambda_param  # Коэффициент регуляризации
        self.n_iters = n_iters  # Количество итераций обучения
        self.w = None  # Веса модели
        self.b = None  # Смещение модели

    # Обучение модели на данных
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape  # Количество объектов и признаков
        self.w = np.zeros(n_features)  # Изначально все веса равны нулю
        self.b = 0.0  # Начальное смещение

        # Повторяем обучение заданное количество раз
        for _ in range(self.n_iters):
            # Обрабатываем каждый объект обучающей выборки
            for i in range(n_samples):
                # Проверяем, правильно ли объект расположен относительно границы
                condition = y[i] * (np.dot(X[i], self.w) + self.b) >= 1

                if condition:
                    # Если отступ достаточный, обновляем веса с учетом регуляризации
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Если отступ недостаточный, корректируем веса и смещение
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - y[i] * X[i])
                    self.b += self.learning_rate * y[i]

    # Вычисление оценок принадлежности объектов к классам
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w) + self.b

    # Предсказание класса: 1 или -1
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)