# Обучение нейронной сети без учителя

## Алгоритм Кохонена
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class KohonenMap:
    def __init__(self, size=5, dimensions=2):
        self.size = size
        self.weights = np.random.rand(size, size, dimensions)

    def train(self, data, learning_rate=0.1, epochs=100):
        # Подготовим дополнительные переменные для масштабирования
        sigma = max(self.size, self.size) / 2.0  # Начальное значение радиуса соседства
        time_constant = epochs / np.log(sigma)  # Временная константа для экспоненциального убывания

        for epoch in range(epochs):
            for x in data:
                # Найти нейрон-победитель
                winner_idx = np.argmin(np.linalg.norm(self.weights - x, axis=2))
                winner_idx = np.unravel_index(winner_idx, (self.size, self.size))

                # Вычисляем радиус соседства и скорость обучения для текущей эпохи
                sigma_current = sigma * np.exp(-epoch / time_constant)
                learning_rate_current = learning_rate * np.exp(-epoch / epochs)

                # Обновить веса победителя и его соседей
                for i in range(self.size):
                    for j in range(self.size):
                        dist = np.sqrt((winner_idx[0] - i) ** 2 + (winner_idx[1] - j) ** 2)
                        if dist <= sigma_current:  # Обновляем только для нейронов внутри радиуса
                            influence = np.exp(-dist ** 2 / (2 * sigma_current ** 2))
                            self.weights[i, j] += learning_rate_current * influence * (x - self.weights[i, j])

    def visualize(self, data, title="Kohonen SOM"):
        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], c='blue', label='Input data', alpha=0.1)  # Визуализация входных данных
        for i in range(self.size):
            for j in range(self.size):
                ax.scatter(self.weights[i, j, 0], self.weights[i, j, 1], color='red')  # Визуализация нейронов
                if i < self.size - 1:
                    ax.plot([self.weights[i, j, 0], self.weights[i + 1, j, 0]],
                            [self.weights[i, j, 1], self.weights[i + 1, j, 1]], 'k-', linewidth=0.5)
                if j < self.size - 1:
                    ax.plot([self.weights[i, j, 0], self.weights[i, j + 1, 0]],
                            [self.weights[i, j, 1], self.weights[i, j + 1, 1]], 'k-', linewidth=0.5)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.1, n_features=2, random_state=0)

# Нормализация данных
min_data = data.min(axis=0)
max_data = data.max(axis=0)
data = (data - min_data) / (max_data - min_data)

# Инициализация карты Кохонена
som = KohonenMap(size=3, dimensions=2)

# Визуализация начальных весов
som.visualize(data, title="Initial Kohonen SOM")

# Обучение
som.train(data, learning_rate=0.1, epochs=100)

# Визуализация результатов после обучения
som.visualize(data, title="Trained Kohonen SOM")
```

![image](https://github.com/user-attachments/assets/bd9dc219-fd16-4deb-84ad-a27ac841c558)

## Алгоритм Хебба
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Функция для обновления весов по правилу Хебба с затуханием
def hebbian_learning_rule(weights, inputs, learning_rate=0.015, decay_rate=0.01):
    for i in range(weights.shape[0]):
        weights[i, :] *= (1 - decay_rate)
    return weights + learning_rate * np.outer(inputs, inputs)

# Инициализация весов
weights = np.random.normal(0, 1, (3, 3))

# Генерация обучающих данных
data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, n_features=3, random_state=0)

# Нормализация данных
min_data = data.min(axis=0)
max_data = data.max(axis=0)
data = (data - min_data) / (max_data - min_data)

# Сохранение начальных весов для визуализации
initial_weights = np.copy(weights)

colors = ['green', 'orange', 'purple']

# Визуализация до обучения
plt.scatter(data[:, 0], data[:, 1], c='blue', label='Input data')
for i, weight in enumerate(initial_weights):
    plt.plot([0, weight[0]], [0, weight[1]], color=colors[i], linewidth=sizes[i] / 20, alpha=0.6)
    plt.scatter(weight[0], weight[1], s=200, c=colors[i], label=f'Initial neuron {i+1}', alpha=1)
plt.title("Hebbian Learning Before Training")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.legend()
plt.grid(True)
plt.show()

# Обучение
epochs = 200  # Количество эпох обучения
for epoch in range(epochs):
    for inputs in data:
        weights = hebbian_learning_rule(weights, inputs)

# Визуализация результатов
plt.scatter(data[:, 0], data[:, 1], c='blue', label='Input data')

# Визуализация окончательных весов как точек с размером, отражающим величину веса
plt.scatter(data[:, 0], data[:, 1], c='blue', label='Input data')
for i, weight in enumerate(weights):
    plt.plot([0, weight[0]], [0, weight[1]], color=colors[i], linewidth=sizes[i] / 20, alpha=0.6)
    plt.scatter(weight[0], weight[1], s=200, c=colors[i], label=f'Trained neuron {i+1}', alpha=1)
plt.title("Hebbian Learning After Training")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Вывод начальных и окончательных весов в текстовом формате
print("Initial weights:\n", initial_weights)
print("Trained weights:\n", weights)
```
![image](https://github.com/user-attachments/assets/9818f255-8023-4716-b9e2-a23dce03f1d5)
