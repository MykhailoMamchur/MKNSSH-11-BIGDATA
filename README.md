# Прогнозування та аналіз факторів успішності фільму за допомогою Big Data та методів машинного навчання​

Проєкт спрямований на використання технологій Big Data для аналізу та передбачення критеріїв успішності фільмів. На основі даних з бази IMDb створено датасет та натреновано модель машинного навчання для класифікації успішності фільму із подальшим визначенням ключових факторів за допомогою SHAP-аналізу.

## Опис результатів

1. **Ефективність Apache Spark**: 
   - Інструмент Apache Spark показав свою високу ефективність при обробці великих обсягів даних, досягнувши обробки в 7 разів швидше на прикладі датасету IMDb, що містить десятки мільйонів рядків і розподілений по різним таблицям.
   
2. **Моделювання з XGBoost**:
   - Модель XGBoost показала вражаючу точність на навчальному наборі даних (98%) та задовільний результат на тестовому наборі (75%). Це відкриває потенціал для подальшого вдосконалення моделі шляхом додавання додаткових даних або оптимізації параметрів моделі.
   
3. **Аналіз SHAP**:
   - Використання SHAP-аналізу дозволило виявити найвпливовіші ознаки для прогнозування успіху фільмів, серед яких виявились такі фактори як тривалість, кількість країн-виробників і жанри. Це підкреслює цінність великих даних для прогнозування успіху фільмів.

## Зміст

1. **Обробка даних і створення нових ознак**
   - `dataset_generate_initial_form`: Обробляє сирі дані фільмів та телешоу, очищає і комбінує різні набори даних.
   - `dataset_add_people_columns`: Додає додаткові дані про сценаристів і режисерів, збагачуючи датасет кількістю відомих фільмів і професій.
   - `dataset_add_popularity_columns`: Додає ознаки популярності режисерів і сценаристів на основі рейтингів і кількості голосів.
   - `dataset_cleanup_columns`: Очищає датасет, видаляючи непотрібні колонки.

2. **Моделювання та оцінка**
   - `train_model`: Навчає модель бінарної класифікації за допомогою XGBoost для прогнозування рейтингів.
   - `score_model`, `score_model_f1`, `score_model_show_cm`: Оцінює точність моделі, F1-міру та побудова матриці плутанини.

3. **Генерація вбудованих представлень (embeddings)**
   - `generate_add_embeddings`: Генерує вбудовані представлення для назв фільмів та телешоу за допомогою моделі `SentenceTransformer`, з подальшим зменшенням розмірності за допомогою UMAP.

4. **Візуалізація**
   - `plot_genres_popularity`: Показує популярність жанрів у часі на основі кількості голосів.
   - `plot_genres_rating`: Показує середні рейтинги жанрів у часі.
   - `plot_genres_interactive`: Створює інтерактивну візуалізацію з можливістю вибору жанру для порівняння трендів популярності та рейтингів.

5. **Утилітні функції**
   - `load_dataset`: Завантажує датасет з файлу Parquet.
   - `preprocess_dataset`: Обробляє датасет, додаючи колонку для категорії рейтингів.
   - `get_splits`: Розділяє датасет на тренувальні та тестові набори для задач машинного навчання.

## Встановлення

Щоб виконати код цього репозиторію, потрібно встановити наступні Python пакети:

```bash
pip install -r requirements.txt
```

Також потрібно додати файли датасету IMDb у папку `data`.

## Використання

```bash
python main.py
```