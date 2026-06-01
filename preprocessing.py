# 1. Import libraries (json, pandas)
import os
import pandas as pd
import json

# 2. Load data/raw/bootstrap.json
with open('data/raw/bootstrap.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
# 3. Pull out the "elements" list into a pandas DataFrame
df=pd.DataFrame(data["elements"])
# print(df.columns.tolist())
# 4. Select only the columns you need
df = df[[
    "first_name", "second_name", "team", "element_type", 
    "now_cost", "total_points", "goals_scored", "assists", 
    "clean_sheets", "minutes", "form", "ict_index",
    "goals_conceded", "saves", "bonus", "selected_by_percent",
    "points_per_game", "expected_goals", "expected_assists"
]]
# 5. Rename or transform any columns (e.g. now_cost → price in millions)
df["now_cost"] = df["now_cost"] / 10
df = df.rename(columns={"now_cost": "price"})
# 6. Print the first 5 rows as a sanity check
print(df.head())

# 7. Save to data/processed/players.csv
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/players.csv",index =False)