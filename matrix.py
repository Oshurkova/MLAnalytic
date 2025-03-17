import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Загрузка данных
print("Загрузка данных...")
df = pd.read_csv("requirements_traceability_dataset_complex.csv")

# Отладочная информация: вывод первых нескольких строк данных
print("Первые несколько строк данных:")
print(df.head())

# Разделение на признаки и целевую переменную
print("Разделение на признаки и целевую переменную...")
X = df["Текст требования"]
y = df["Связанные элементы проекта"]

# Векторизация текстов требований
vectorizer = TfidfVectorizer(max_features=1000)
X_vectorized = vectorizer.fit_transform(X)

# Разделение на обучающую и тестовую выборки
print("Разделение на обучающую и тестовую выборки...")
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Обучение модели классификации
print("Обучение модели классификации...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка модели на тестовой выборке
print("Оценка модели на тестовой выборке...")
y_pred = model.predict(X_test)

# Расчет метрик
logistic_accuracy = accuracy_score(y_test, y_pred)
logistic_precision = precision_score(y_test, y_pred, average='weighted')
logistic_recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {logistic_accuracy}")
print(f"Precision: {logistic_precision}")
print(f"Recall: {logistic_recall}")

# Пример предсказания связей для новых требований
new_requirements = [
    "Система должна иметь возможность отправлять уведомления о статусе заказа.",
    "Система должна предоставлять возможность оплаты товаров через различные платежные системы."
]

# Векторизация новых требований
new_requirements_vectorized = vectorizer.transform(new_requirements)

# Предсказание связей для новых требований
predicted_links = model.predict(new_requirements_vectorized)
for req, links in zip(new_requirements, predicted_links):
    print(f"Требование: {req}")
    print(f"Предполагаемые связанные элементы проекта: {links}")
    print()

