import pandas as pd
import re
from metrics import ValidatorSimple  # Ваш код метрик
from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd

# Читаем данные
train_path = "/home/guest/Desktop/project/hackathon_hse25/prepocess_calculate/datasets/train_set.json"
val_path = "/home/guest/Desktop/project/hackathon_hse25/prepocess_calculate/datasets/val_set.json"

train_df = pd.read_json(train_path) 
val_df = pd.read_json(val_path)

combined_df = pd.concat([train_df, val_df], ignore_index=True)

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

# Проверка
for i in range(min(5, len(processed_df))):
    print(f"Question: {processed_df.iloc[i]['question']}")
    print(f"Answer: {processed_df.iloc[i]['answer']}")
    print(f"Ground Truth: {processed_df.iloc[i]['ground_truth']}")
    print(f"Context: {processed_df.iloc[i]['context']}")
    print("-" * 50)
app = FastAPI()
class UserFilter(BaseModel):
    user: str  # Фильтрация по пользователю

@app.get("/filter/")
async def filter_by_city(user_filter: UserFilter = Query(None)):  # Параметр по умолчанию None
    if user_filter and user_filter.user:  # Если user_filter передан
        # Фильтрация по городу
        user_city = combined_df[combined_df['Кампус'] == user_filter.user]
    else:
        # Без фильтрации, возвращаем все данные
        user_city = combined_df

    if user_city.empty:
        return {"message": "No data found for the user."}
    
    # Возвращаем отфильтрованные данные
    return {"filtered_data": user_city[['Ответ AI', 'Ресурсы для ответа']].to_dict(orient="records")}

# Валидация
validator = ValidatorSimple(neural=True)
results, mid_percentage, good_percentage, bad_percentage = validator.validate_rag(processed_df)
print("Результаты валидации:")
print(results)
print(mid_percentage)
print(good_percentage)
print(bad_percentage)