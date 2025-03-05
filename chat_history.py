import pandas as pd
import re
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка данных
train_path = "/workspaces/hackathon_hse25/prepocess_calculate/datasets/train_set.json"
val_path = "/workspaces/hackathon_hse25/prepocess_calculate/datasets/val_set.json"

train_df = pd.read_json(train_path)
val_df = pd.read_json(val_path)

# Объединение данных
combined_df = pd.concat([train_df, val_df], ignore_index=True)

# Функция предобработки текста
def preprocess_text(text):
    text = text.lower()  # Приведение к нижнему регистру
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации
    return text

# Применение предобработки
combined_df['processed_text'] = combined_df['Вопрос пользователя'].apply(preprocess_text)

# Инициализация модели SentenceTransformer
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

# Вычисление эмбеддингов
texts = combined_df['processed_text'].tolist()
embeddings = model.encode(texts)

# Вычисление матрицы сходства
similarity_matrix = cosine_similarity(embeddings)

# Группировка вопросов
threshold = 0.8  # Порог семантического сходства
used_indices = set()
groups = {}

for i in range(len(texts)):
    if i in used_indices:
        continue  # Пропускаем, если вопрос уже в группе

    groups[i] = [i]  # Создаём новую группу
    for j in range(i + 1, len(texts)):
        if j not in used_indices and similarity_matrix[i, j] > threshold:
            groups[i].append(j)
            used_indices.add(j)

# Подсчёт частоты групп
group_frequencies = {texts[i]: len(indices) for i, indices in groups.items()}

# Вывод результатов
print("Частота повторяющихся вопросов:")
for question, count in group_frequencies.items():
    print(f"Группа: {question}, Количество повторений: {count}")

# Создание словаря для хранения групп и их категорий
group_categories = defaultdict(list)

# Определение категорий для каждой группы
for group_idx, question_indices in groups.items():
    representative_question = texts[group_idx]  # Представитель группы
    categories_in_group = combined_df.iloc[question_indices]['Категория вопроса'].tolist()
    group_categories[representative_question] = categories_in_group

# Подсчёт частоты категорий в группах
category_frequencies = defaultdict(int)
for categories in group_categories.values():
    for category in categories:
        category_frequencies[category] += 1

# Вывод результатов
print("Частота категорий в группах:")
for category, frequency in category_frequencies.items():
    print(f"Категория: {category}, Частота: {frequency}")

chat_history_df = pd.DataFrame(category_frequencies.items(), columns=['Категория', 'Частота'])
chat_history_df.to_csv('category_frequency.csv', index=False, encoding='utf-8')
