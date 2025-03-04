from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd

# Создаем экземпляр FastAPI
app = FastAPI()

train_path = "/home/guest/Desktop/project/hackathon_hse25/prepocess_calculate/datasets/train_set.json"
val_path = "/home/guest/Desktop/project/hackathon_hse25/prepocess_calculate/datasets/val_set.json"

train_df = pd.read_json(train_path) 
val_df = pd.read_json(val_path)

combined_df = pd.concat([train_df, val_df], ignore_index=True)

processed_df = pd.DataFrame(combined_df)

class UserFilter(BaseModel):
    user: str  # Фильтрация по пользователю

@app.get("/filter/")
async def filter_by_user(user_filter: UserFilter):
    # Фильтрация DataFrame по имени пользователя
    user_city = processed_df[processed_df['Кампус'] == user_filter.user]
    
    if user_city.empty:
        return {"message": "No data found for the user."}
    
    # Возвращаем отфильтрованные данные (например, answer и context)
    result = user_city[['answer', 'context']].to_dict(orient="records")
    return {"filtered_data": result}
