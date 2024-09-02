import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = pd.read_csv('corn.csv', sep=';')
x = data['wavelength'].values
Y = data['Spectr'].values

# Масштабирование данных
scaler_x = StandardScaler()
scaler_Y = StandardScaler()

x_scaled = scaler_x.fit_transform(x.reshape(-1, 1)).flatten()
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()

# 1. Аналитическое вычисление коэффициентов
n = len(x_scaled)
x_mean = np.mean(x_scaled)
Y_mean = np.mean(Y_scaled)
x_squared_mean = np.mean(x_scaled**2)
xy_mean = np.mean(x_scaled * Y_scaled)

a1_analytic = (n * xy_mean - np.sum(x_scaled) * np.sum(Y_scaled)) / (n * x_squared_mean - (np.sum(x_scaled))**2)
a0_analytic = Y_mean - a1_analytic * x_mean

# Построение графика аналитической регрессии
x_space_scaled = np.linspace(x_scaled.min(), x_scaled.max(), 100)
Y_pred_analytic = a0_analytic + a1_analytic * x_space_scaled

# 2. Коэффициенты с использованием метода градиентного спуска
class SimpleRegressionGD:
    def __init__(self):
        self.a0 = np.random.rand()
        self.a1 = np.random.rand()

    def predict(self, x):
        return self.a0 + self.a1 * x

    def MSE(self, x, Y):
        predictions = self.predict(x)
        return np.nan_to_num(((Y - predictions) ** 2).mean())

    def fit(self, x, Y, alpha=0.00001, epsilon=0.01, max_steps=5000):
        steps, errors = [], []
        for step in range(max_steps):
            predictions = self.predict(x)
            dT_a0 = -2 * np.sum(Y - predictions)
            dT_a1 = -2 * np.sum((Y - predictions) * x)
            self.a0 -= alpha * dT_a0
            self.a1 -= alpha * dT_a1
            new_error = self.MSE(x, Y)
            if np.isnan(new_error) or np.isinf(new_error):
                print(f"Error is NaN or Inf at step {step}")
                break
            steps.append(step)
            errors.append(new_error)
            if new_error < epsilon:
                break
        return steps, errors

# Запуск градиентного спуска
regr_gd = SimpleRegressionGD()
steps, errors = regr_gd.fit(x_scaled, Y_scaled)

# Проверка результатов градиентного спуска
print(f'Коэффициенты после градиентного спуска: a0 = {regr_gd.a0}, a1 = {regr_gd.a1}')
print(f'Последний MSE: {regr_gd.MSE(x_scaled, Y_scaled)}')

# Прогнозы с использованием градиентного спуска
Y_pred_gd = regr_gd.predict(x_space_scaled)

# Обратное масштабирование для восстановления исходного масштаба
x_space = scaler_x.inverse_transform(x_space_scaled.reshape(-1, 1)).flatten()
Y_pred_analytic = scaler_Y.inverse_transform(Y_pred_analytic.reshape(-1, 1)).flatten()
Y_pred_gd = scaler_Y.inverse_transform(Y_pred_gd.reshape(-1, 1)).flatten()

# 3. Построение графиков
plt.figure(figsize=(14, 7))

# График аналитической регрессии
plt.subplot(1, 2, 1)
plt.scatter(x, Y, color='blue', label='Исходные данные')
plt.plot(x_space, Y_pred_analytic, color='red', label='Аналитическая регрессия')
plt.title('Аналитическая регрессия')
plt.xlabel('Длина волны')
plt.ylabel('Спектр')
plt.legend()

# График регрессии градиентного спуска
plt.subplot(1, 2, 2)
plt.scatter(x, Y, color='blue', label='Исходные данные')
plt.plot(x_space, Y_pred_gd, color='green', label='Регрессия градиентного спуска')
plt.title('Регрессия градиентного спуска')
plt.xlabel('Длина волны')
plt.ylabel('Спектр')
plt.legend()

plt.tight_layout()
plt.show()
