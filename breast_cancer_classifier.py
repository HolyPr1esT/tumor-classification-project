# 1. Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Установим красивый стиль для графиков Seaborn
sns.set_theme(style="whitegrid")

# 2. Загрузка датасета
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target # 0 - malignant (злокачественная), 1 - benign (доброкачественная)

# 3. Анализ данных (EDA - Exploratory Data Analysis)
print("Первые 5 строк данных:")
print(df.head())
print("\nИнформация о датасете:")
df.info()
print("\nСтатистическое описание данных:")
print(df.describe().T) # .T для транспонирования, чтобы было удобнее читать
print("\nРаспределение целевой переменной:")
print(df['target'].value_counts())
sns.countplot(x='target', data=df)
plt.title('Распределение классов (0=Malignant, 1=Benign)')
plt.xticks([0, 1], ['Злокачественная', 'Доброкачественная'])
plt.show()

# Корреляционная матрица (только первые 10 признаков для наглядности + target)
# В реальном проекте можно анализировать всё или выбирать группы признаков
corr_matrix = df.iloc[:, :10].join(df['target']).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Корреляционная матрица (первые 10 признаков и target)')
plt.show()

# 4. Подготовка данных
X = df.drop('target', axis=1)
y = df['target']

# Разделение на обучающую и тестовую выборки
# stratify=y важен для сохранения пропорций классов в обеих выборках
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nРазмер обучающей выборки X: {X_train.shape}")
print(f"Размер тестовой выборки X: {X_test.shape}")

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Используем transform, а не fit_transform для тестовой выборки!

# 5. Создание и обучение моделей
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=10000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42, probability=True) # probability=True для возможности ROC-AUC
}

results = {}

for model_name, model in models.items():
    print(f"\n--- Обучение модели: {model_name} ---")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=cancer.target_names)
    cm = confusion_matrix(y_test, y_pred)
    
    results[model_name] = {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "model_object": model # Сохраняем обученную модель
    }
    
    print(f"Точность (Accuracy): {accuracy:.4f}")
    print("Отчет по классификации:")
    print(report)
    
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cancer.target_names, yticklabels=cancer.target_names)
    plt.xlabel("Предсказанные классы")
    plt.ylabel("Истинные классы")
    plt.title(f'Матрица ошибок: {model_name}')
    plt.show()

# 6. Выбор лучшей модели (по Accuracy, но можно и по другим метрикам, например, Recall для класса 'malignant')
best_model_name = max(results, key=lambda name: results[name]['accuracy'])
best_model_metrics = results[best_model_name]
best_model_object = best_model_metrics['model_object']

print(f"\n--- Лучшая модель: {best_model_name} ---")
print(f"Точность: {best_model_metrics['accuracy']:.4f}")
print("Отчет:")
print(best_model_metrics['report'])

# 7. (Опционально) Подбор гиперпараметров для лучшей модели (например, Random Forest)
if best_model_name == "Random Forest":
    print("\n--- Подбор гиперпараметров для Random Forest с GridSearchCV ---")
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # В реальном проекте CV лучше ставить 5 или 10, для скорости здесь 3
    # n_jobs=-1 использует все доступные ядра процессора
    grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search_rf.fit(X_train_scaled, y_train)
    
    print("Лучшие параметры для Random Forest:", grid_search_rf.best_params_)
    best_rf_tuned = grid_search_rf.best_estimator_ # Это уже обученная модель с лучшими параметрами
    
    y_pred_rf_tuned = best_rf_tuned.predict(X_test_scaled)
    accuracy_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)
    report_rf_tuned = classification_report(y_test, y_pred_rf_tuned, target_names=cancer.target_names)
    cm_rf_tuned = confusion_matrix(y_test, y_pred_rf_tuned)

    print(f"\nТочность Random Forest (после GridSearchCV): {accuracy_rf_tuned:.4f}")
    print("Отчет по классификации (после GridSearchCV):")
    print(report_rf_tuned)
    
    plt.figure(figsize=(6,4))
    sns.heatmap(cm_rf_tuned, annot=True, fmt="d", cmap="Blues", xticklabels=cancer.target_names, yticklabels=cancer.target_names)
    plt.xlabel("Предсказанные классы")
    plt.ylabel("Истинные классы")
    plt.title(f'Матрица ошибок: Random Forest (Tuned)')
    plt.show()
    
    # Обновим лучшую модель, если она действительно стала лучше
    if accuracy_rf_tuned > best_model_metrics['accuracy']:
        print("Random Forest после тюнинга показал лучшие результаты!")
        best_model_name += " (Tuned)"
        results[best_model_name] = {
            "accuracy": accuracy_rf_tuned,
            "report": report_rf_tuned,
            "confusion_matrix": cm_rf_tuned,
            "model_object": best_rf_tuned
        }

print("\n--- Финальные результаты ---")
for name, res in results.items():
    print(f"Модель: {name}, Точность: {res['accuracy']:.4f}")

