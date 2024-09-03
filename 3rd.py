import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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

# Создание и обучение модели kNN
k = 5  # Выбор количества соседей
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test_scaled)

# Вывод матрицы путаницы
conf_matrix = confusion_matrix(y_test, y_pred)
print("Матрица путаницы:")
print(conf_matrix)

# Визуализация матрицы путаницы
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Не высокий', 'Высокий'], yticklabels=['Не высокий', 'Высокий'])
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица путаницы')
plt.show()

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")

# Вывод отчета по классификации
print("Отчет по классификации:")
print(classification_report(y_test, y_pred, target_names=['Не высокий', 'Высокий']))
