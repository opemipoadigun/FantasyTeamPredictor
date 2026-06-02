# 1. Import libraries (json, pandas)
import os
import pandas as pd
import json

# 2. Load data/raw/bootstrap.json
with open('data/raw/bootstrap.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
with open('data/raw/gw_history.json', 'r', encoding='utf-8') as file:
    history = json.load(file)
with open('data/raw/past_history.json', 'r', encoding='utf-8') as file:
    past_history = json.load(file)
# 3. Pull out the "elements" list into a pandas DataFrame
df=pd.DataFrame(data["elements"])
gw_df = pd.DataFrame(history)
past_df = pd.DataFrame(past_history)
# print(df.columns.tolist())
# 4. Select only the columns you need
df = df[[
    "first_name", "second_name", "team", "element_type", "now_cost", "id","code"]]
# 5. Rename or transform any columns (e.g. now_cost → price in millions)
df["now_cost"] = df["now_cost"] / 10
df = df.rename(columns={"now_cost": "price"})
past_df["total_points_per_game"] = past_df["total_points"] / (past_df["minutes"] / 90)
past_df["goals_scored_per_game"] = past_df["goals_scored"] / (past_df["minutes"] / 90)
past_df["assists_per_game"] = past_df["assists"] / (past_df["minutes"] / 90)
past_df["saves_per_game"] = past_df["saves"] / (past_df["minutes"] / 90)
past_df["bonus_per_game"] = past_df["bonus"] / (past_df["minutes"] / 90)
past_df["clean_sheets_per_game"] = past_df["clean_sheets"] / (past_df["minutes"] / 90)
past_df["yellow_cards_per_game"] = past_df["yellow_cards"] / (past_df["minutes"] / 90)
past_df["red_cards_per_game"] = past_df["red_cards"] / (past_df["minutes"] / 90)
past_df["goals_conceded_per_game"] = past_df["goals_conceded"] / (past_df["minutes"] / 90)
past_df["penalties_saved_per_game"] = past_df["penalties_saved"] / (past_df["minutes"] / 90)
past_df["penalties_missed_per_game"] = past_df["penalties_missed"] / (past_df["minutes"] / 90)

merged_df = pd.merge(df, gw_df, left_on="id", right_on="element")
merged_df = merged_df.sort_values(by=["element","round"])

merged_df["next_gw_points"] = merged_df.groupby("element")["total_points"].transform(lambda x: x.shift(-1))

merged_df["total_points"] = merged_df.groupby("element")["total_points"].transform(lambda x: x.rolling(3).mean())
merged_df["minutes"] = merged_df.groupby("element")["minutes"].transform(lambda x: x.rolling(3).mean())
merged_df["goals_scored"] = merged_df.groupby("element")["goals_scored"].transform(lambda x: x.rolling(3).mean())
merged_df["assists"] = merged_df.groupby("element")["assists"].transform(lambda x: x.rolling(3).mean())
merged_df["saves"] = merged_df.groupby("element")["saves"].transform(lambda x: x.rolling(3).mean())
merged_df["bonus"] = merged_df.groupby("element")["bonus"].transform(lambda x: x.rolling(3).mean())
merged_df["clean_sheets"] = merged_df.groupby("element")["clean_sheets"].transform(lambda x: x.rolling(3).mean())
merged_df["yellow_cards"] = merged_df.groupby("element")["yellow_cards"].transform(lambda x: x.rolling(3).mean())
merged_df["red_cards"] = merged_df.groupby("element")["red_cards"].transform(lambda x: x.rolling(3).mean())
merged_df["goals_conceded"] = merged_df.groupby("element")["goals_conceded"].transform(lambda x: x.rolling(3).mean())
merged_df["penalties_saved"] = merged_df.groupby("element")["penalties_saved"].transform(lambda x: x.rolling(3).mean())
merged_df["penalties_missed"] = merged_df.groupby("element")["penalties_missed"].transform(lambda x: x.rolling(3).mean())

merged_df = merged_df.dropna(subset=["next_gw_points"])
# 6. Print the first 5 rows as a sanity check
print(merged_df.head())
# 7. Save to data/processed/players.csv
os.makedirs("data/processed", exist_ok=True)
merged_df.to_csv("data/processed/gw_players.csv",index =False)