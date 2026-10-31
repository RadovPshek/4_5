import numpy as np
from typing import Callable, Optional


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    # Обычное евклидово расстояние: корень из суммы квадратов разностей координат
    return float(np.sqrt(np.sum((x - y) ** 2)))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    # Манхэттенское расстояние: сумма модулей разностей координат
    return float(np.sum(np.abs(x - y)))


class KNNClassifier:
    def __init__(self, k: int = 3, metric: Optional[Callable] = None, weights: str = 'uniform'):
        self.k = k

        # Если метрика не передана - по умолчанию используем евклидову.
        # Так вызывающему не нужно каждый раз передавать её явно.
        self.metric = metric if metric is not None else euclidean_distance

        # Способ взвешивания голосов: 'uniform' - все соседи равны,
        # любое другое значение трактуем как 'distance' - вес обратно пропорционален расстоянию
        self.weights = weights

        # Обучающая выборка заполняется в fit(); до обучения здесь None -
        # это используется как индикатор «модель не обучена»
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # KNN - ленивый алгоритм: само обучение сводится к запоминанию данных.
        # Вся работа происходит в predict().
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Защита от вызова predict() до fit():
        # без обучающих меток предсказывать нечего, возвращаем пустой массив
        if self.y_train is None:
            return np.array([])

        predictions = []

        # Классифицируем каждую точку независимо
        for x in X:
            # Считаем расстояние от точки до всех обучающих примеров
            distances = np.array([self.metric(x, x_train) for x_train in self.X_train])

            # Индексы k ближайших соседей: argsort сортирует по возрастанию,
            # берём первые k
            k_indices = np.argsort(distances)[:self.k]

            # Метки и расстояния этих соседей - понадобятся для голосования
            k_labels = self.y_train[k_indices]
            k_distances = distances[k_indices]

            if self.weights == 'uniform':
                # Простое голосование: побеждает класс, который встречается чаще.
                # При равенстве голосов np.argmax вернёт первый по порядку -
                # это детерминированно, но не «честно»; для базового KNN приемлемо.
                classes, counts = np.unique(k_labels, return_counts=True)
                predictions.append(classes[np.argmax(counts)])
            else:  # distance
                # Взвешенное голосование: ближние соседи весят больше.
                # Деление на ноль защищаем: если расстояние 0 (совпадающая точка),
                # даём ей огромный вес 1e10, чтобы она гарантированно победила.
                weights = np.where(k_distances == 0, 1e10, 1.0 / k_distances)

                # Суммируем веса по каждому классу вручную:
                # numpy не умеет группировать по произвольным меткам одной командой
                class_weights = {}
                for label, w in zip(k_labels, weights):
                    class_weights[label] = class_weights.get(label, 0) + w

                # Берём класс с максимальной суммарной «массой» голосов
                predictions.append(max(class_weights, key=class_weights.get))

        return np.array(predictions)