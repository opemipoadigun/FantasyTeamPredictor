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

lastseason_df = pd.read_csv('data/raw/lastseason.csv')

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

merged_df = pd.merge(df, gw_df, left_on="id", right_on="element")
merged_df = merged_df.sort_values(by=["element","round"])

merged_df["next_gw_points"] = merged_df.groupby("element")["total_points"].transform(lambda x: x.shift(-1))
merged_df["total_points"] = merged_df.groupby("element")["total_points"].transform(lambda x: x.rolling(3, min_periods=1).mean())
merged_df["minutes"] = merged_df.groupby("element")["minutes"].transform(lambda x: x.rolling(3, min_periods=1).mean())
merged_df["goals_scored"] = merged_df.groupby("element")["goals_scored"].transform(lambda x: x.rolling(3, min_periods=1).mean())
merged_df["assists"] = merged_df.groupby("element")["assists"].transform(lambda x: x.rolling(3, min_periods=1).mean())
merged_df["saves"] = merged_df.groupby("element")["saves"].transform(lambda x: x.rolling(3, min_periods=1).mean())
merged_df["bonus"] = merged_df.groupby("element")["bonus"].transform(lambda x: x.rolling(3, min_periods=1).mean())
merged_df["clean_sheets"] = merged_df.groupby("element")["clean_sheets"].transform(lambda x: x.rolling(3, min_periods=1).mean())
merged_df["yellow_cards"] = merged_df.groupby("element")["yellow_cards"].transform(lambda x: x.rolling(3, min_periods=1).mean())
merged_df["red_cards"] = merged_df.groupby("element")["red_cards"].transform(lambda x: x.rolling(3, min_periods=1).mean())
merged_df["goals_conceded"] = merged_df.groupby("element")["goals_conceded"].transform(lambda x: x.rolling(3, min_periods=1).mean())
merged_df["penalties_saved"] = merged_df.groupby("element")["penalties_saved"].transform(lambda x: x.rolling(3, min_periods=1).mean())
merged_df["penalties_missed"] = merged_df.groupby("element")["penalties_missed"].transform(lambda x: x.rolling(3, min_periods=1).mean())

merged_df = merged_df.dropna(subset=["next_gw_points"])
merged_df["season"] = "2026-27"
# 6. Print the first 5 rows as a sanity check
print(merged_df.head())
# 7. Save to data/processed/players.csv
os.makedirs("data/processed", exist_ok=True)
# merged_df.to_csv("data/processed/gw_players.csv",index =False)

position_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
lastseason_df["element_type"] = lastseason_df["position"].map(position_map)
lastseason_df["value"] = lastseason_df["value"] /10
lastseason_df = lastseason_df.rename(columns={"value":"price"})
lastseason_df = lastseason_df.sort_values(["element", "round"])

lastseason_df["next_gw_points"] = lastseason_df.groupby("element")["total_points"].transform(lambda x: x.shift(-1))
lastseason_df["total_points"] = lastseason_df.groupby("element")["total_points"].transform(lambda x: x.rolling(3, min_periods=1).mean())
lastseason_df["minutes"] = lastseason_df.groupby("element")["minutes"].transform(lambda x: x.rolling(3, min_periods=1).mean())
lastseason_df["goals_scored"] = lastseason_df.groupby("element")["goals_scored"].transform(lambda x: x.rolling(3, min_periods=1).mean())
lastseason_df["assists"] = lastseason_df.groupby("element")["assists"].transform(lambda x: x.rolling(3, min_periods=1).mean())
lastseason_df["saves"] = lastseason_df.groupby("element")["saves"].transform(lambda x: x.rolling(3, min_periods=1).mean())
lastseason_df["bonus"] = lastseason_df.groupby("element")["bonus"].transform(lambda x: x.rolling(3, min_periods=1).mean())
lastseason_df["clean_sheets"] = lastseason_df.groupby("element")["clean_sheets"].transform(lambda x: x.rolling(3, min_periods=1).mean())
lastseason_df["yellow_cards"] = lastseason_df.groupby("element")["yellow_cards"].transform(lambda x: x.rolling(3, min_periods=1).mean())
lastseason_df["red_cards"] = lastseason_df.groupby("element")["red_cards"].transform(lambda x: x.rolling(3, min_periods=1).mean())
lastseason_df["goals_conceded"] = lastseason_df.groupby("element")["goals_conceded"].transform(lambda x: x.rolling(3, min_periods=1).mean())
lastseason_df["penalties_saved"] = lastseason_df.groupby("element")["penalties_saved"].transform(lambda x: x.rolling(3, min_periods=1).mean())
lastseason_df["penalties_missed"] = lastseason_df.groupby("element")["penalties_missed"].transform(lambda x: x.rolling(3, min_periods=1).mean())
lastseason_df["season"] = "2025-26"

feature_cols = ["total_points", "minutes", "goals_scored", "assists", "saves", "bonus", "clean_sheets", "yellow_cards", "red_cards", "goals_conceded", "penalties_saved", "penalties_missed", "element_type", "price"]
lastseason_df = lastseason_df.dropna(subset=feature_cols)

lastseason_df = lastseason_df.dropna(subset=["next_gw_points"])
lastseason_df["first_name"] = lastseason_df["name"]
lastseason_df["second_name"] = ""

combined_df = pd.concat([merged_df,lastseason_df])

os.makedirs("data/processed", exist_ok=True)
combined_df.to_csv("data/processed/combined_data.csv", index=False)

print(f"Combined rows: {len(combined_df)}")
