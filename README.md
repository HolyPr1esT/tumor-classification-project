# Проект: Классификация опухолей молочной железы с использованием машинного обучения  
**Автор**: Малиновский Антон Валерьевич, МИК21

## Описание проекта  
Данный проект нацелен на разработку модели машинного обучения для классификации опухолей молочной железы как злокачественных (*malignant*) или доброкачественных (*benign*).  
В основе лежат физико-химические и морфологические характеристики клеточных ядер, полученные из оцифрованных изображений тонкоигольной аспирационной биопсии (FNA).  
Точная и своевременная диагностика рака молочной железы критически важна для успешного лечения, и модели машинного обучения могут служить вспомогательным инструментом для врачей.

В репозитории доступна вкладка с Github Actions, там результат работы кода
---

## Использованные данные  

- **Датасет**: Breast Cancer Wisconsin (Diagnostic)  
- **Источник**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
  *(Dr. William H. Wolberg, University of Wisconsin Hospitals, Madison)*  
- **Количество образцов**: 569  
- **Количество признаков**: 30 числовых признаков, вычисленных для каждого клеточного ядра  
- **Примеры признаков**:  
  - `mean radius` (средний радиус)  
  - `mean texture` (средняя текстура)  
  - `mean perimeter` (средний периметр)  
  - `mean area` (средняя площадь)  
  - `mean smoothness` (средняя гладкость)  
  - ...и аналогичные характеристики для стандартной ошибки (*SE*) и "худших" (*worst*) значений  
- **Целевая переменная**: `target`  
  - `0`: Malignant (злокачественная)  
  - `1`: Benign (доброкачественная)  

---

## Предобработка данных  

- **Загрузка и инспекция**:  
  Данные загружены и проанализированы на предмет структуры, типов данных и пропущенных значений (отсутствуют).

- **Визуализация**:  
  Исследовано распределение целевой переменной и корреляции между признаками.

- **Разделение выборки**:  
  75% — обучающая выборка  
  25% — тестовая выборка  
  С применением стратификации (`stratify=y`) и `random_state=42` для воспроизводимости.

- **Масштабирование признаков**:  
  Применён `StandardScaler` для нормализации данных, что улучшает качество моделей, чувствительных к масштабу.

---

## Модели машинного обучения  

Обучены и протестированы следующие алгоритмы:

1. **Логистическая регрессия (Logistic Regression)**  
   - Простая и интерпретируемая модель  
2. **Случайный лес (Random Forest Classifier)**  
   - Ансамблевый метод, устойчивый к переобучению  
3. **Метод опорных векторов (Support Vector Machine, SVC)**  
   - Эффективен при высокой размерности признаков  

---

## Оценка моделей  

Модели оценивались по следующим метрикам:

- **Accuracy (точность)** — доля правильных предсказаний  
- **Classification Report** — включает:
  - Precision (точность предсказаний)  
  - Recall (полнота)  
  - F1-score (среднее гармоническое precision и recall)  
- **Confusion Matrix (матрица ошибок)** — количество:
  - TP — истинно положительных  
  - TN — истинно отрицательных  
  - FP — ложноположительных  
  - FN — ложноотрицательных  

> ⚠ Особое внимание уделялось **Recall для класса "malignant"**, так как пропуск злокачественной опухоли может быть опасным.

---

## Результаты  

| Модель                | Accuracy       |
|----------------------|----------------|
| Логистическая регрессия | ~97.2%         |
| Случайный лес          | ~95.8% (до тюнинга) |
| SVC                    | ~97.9%         |

- **SVC** показала наилучшие результаты до оптимизации.
- Для **Random Forest** применён `GridSearchCV` для подбора гиперпараметров:
  - После оптимизации: **~97.9% – 98.6%**

---

## ✅ Итоги проекта  

- Разработаны и протестированы три ML-модели для классификации опухолей.  
- Все модели достигли **точности выше 95%**.  
- **Наилучшие результаты** показали:
  - **SVC**
  - **Random Forest после тюнинга**
- Ключевой метрикой при оценке являлась **Recall для "malignant"**.
- Подтверждена важность:
  - Предобработки данных (масштабирование)
  - Подбора гиперпараметров

---

## Ограничения  

- Модель обучена на одном датасете — производительность на других данных может отличаться.  
- Модель **не заменяет** профессиональное медицинское заключение.

---

## Перспективы развития  

- **Feature Engineering / Selection**: Создание и отбор признаков  
- **Продвинутые модели**:  
  - XGBoost  
  - LightGBM  
  - Нейросети  
- **Интерпретируемость**:  
  - SHAP  
  - LIME  
- **Кросс-валидация**: Более строгая оценка моделей  
- **Развертывание**:  
  - Веб-приложение  
  - REST API  
  *(с предупреждением об использовании только в качестве вспомогательного инструмента)*

---

## Источники  

- Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1992). Breast cancer Wisconsin (diagnostic) data set.  
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
- [Scikit-learn Documentation](https://scikit-learn.org/)
