import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загрузка данных
data = pd.read_csv('winequality-red.csv')

# Преобразование меток качества в бинарные значения
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Определение признаков и целевого признака
X = data.drop('quality', axis=1)  # Признаки
y = data['quality']  # Целевой признак

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создание и обучение модели
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Получение коэффициентов модели
weights = model.coef_[0]
intercept = model.intercept_[0]

# Вывод коэффициентов и уравнения гиперповерхности
print(f"Массив весов: {weights}")
print(f"Смещение: {intercept}")

print("Уравнение гиперповерхности в общем виде:")
features = [f'feature_{i+1}' for i in range(len(weights))]
equation_terms = [f'{weights[i]:.2f}*{features[i]}' for i in range(len(weights))]
equation = ' + '.join(equation_terms) + f' + {intercept:.2f} = 0'
print(f"Уравнение: {equation}")

# Предсказание и оценка точности модели
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Точность модели на тестовой выборке: {accuracy:.2f}")
