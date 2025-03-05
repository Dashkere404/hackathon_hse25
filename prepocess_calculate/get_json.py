import pandas as pd
import re
from metrics import ValidatorSimple  # Ваш код метрик
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Any

# Читаем данные
val_path = "/workspaces/hackathon_hse25/prepocess_calculate/datasets/val_set.json"
val_df = pd.read_json(val_path)
combined_df = val_df
# Объединяем данные


# Ground truth
def get_ground_truth(row):
    if row["Кто лучше?"] == "Saiga":
        return row["Saiga"]
    elif row["Кто лучше?"] == "Giga":
        return row["Giga"]
    elif row["Кто лучше?"] == "Оба хорошо":
        return row["Saiga"]
    elif row["Кто лучше?"] == "Оба плохо":
        return None
    else:
        return None

combined_df["ground_truth"] = combined_df.apply(get_ground_truth, axis=1)
combined_df = combined_df.dropna(subset=["ground_truth"])

# Processed_df
processed_df = pd.DataFrame({
    "answer": combined_df["Ответ AI"],
    "ground_truth": combined_df["ground_truth"],
    "context": combined_df["Ресурсы для ответа"],
    "question": combined_df["Вопрос пользователя"]
})

# Парсинг контекста
def extract_page_content(context):
    if isinstance(context, str):
        matches = re.findall(r"page_content='(.*?)'", context, re.DOTALL)
        return matches if matches else [""]
    return [""]

# Фильтрация контекста
def filter_context(context, ground_truth):
    filtered = []
    ground_words = set(ground_truth.lower().split())
    for c in context:
        if any(word in c.lower() for word in ground_words):
            filtered.append(c)
    return filtered if filtered else context

processed_df["context"] = processed_df["context"].apply(extract_page_content)
processed_df["context"] = processed_df.apply(
    lambda row: filter_context(row["context"], row["ground_truth"]), axis=1
)

processed_df["answer"] = processed_df["answer"].fillna("")
processed_df["ground_truth"] = processed_df["ground_truth"].fillna("")



# Валидация
validator = ValidatorSimple(neural=True)
results, mid_percentage, good_percentage, bad_percentage = validator.validate_rag(processed_df)

metrics1_dict = {
    "хорошая": good_percentage['context_recall'],
    "плохая": bad_percentage['context_recall'],
    "средняя": mid_percentage['context_recall']
}

metrics2_dict = {
    "хорошая": good_percentage['context_precision'],
    "плохая": bad_percentage['context_precision'],
    "средня": mid_percentage['context_precision']
}

metrics3_dict = {
    "хорошая": good_percentage['answer_correctness_literal'],
    "плохая": bad_percentage['answer_correctness_literal'],
    "средняя": mid_percentage['answer_correctness_literal']
}

metrics4_dict = {
    "хорошая": good_percentage['answer_correctness_neural'],
    "плохая": bad_percentage['answer_correctness_neural'],
    "средняя": mid_percentage['answer_correctness_neural']
}

results_df = pd.DataFrame(results.items())
results_df.to_csv('results.csv', index=False, encoding='utf-8')
metric1=pd.DataFrame(metrics1_dict.items(), columns=['Модель', 'Значение']).to_csv('metrics1.csv', index=False, encoding='utf-8')
metric2=pd.DataFrame(metrics2_dict.items(), columns=['Модель', 'Значение']).to_csv('metrics2.csv', index=False, encoding='utf-8')
metric3=pd.DataFrame(metrics3_dict.items(), columns=['Модель', 'Значение']).to_csv('metrics3.csv', index=False, encoding='utf-8')
metric3=pd.DataFrame(metrics4_dict.items(), columns=['Модель', 'Значение']).to_csv('metrics4.csv', index=False, encoding='utf-8')

# Сохранение объединенного DataFrame в один CSV-файл
combined_df.to_csv('combined_results.csv', index=False, encoding='utf-8')
