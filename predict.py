import pandas as pd
import joblib
import json

model = joblib.load("models/model.joblib")
df = pd.read_csv("data/processed/combined_data.csv", low_memory=False)

with open("data/raw/bootstrap.json", "r", encoding="utf-8") as f:
    bootstrap = json.load(f)

team_map = {t["id"]: t["name"] for t in bootstrap["teams"]}
price_map = {p["id"]: p["now_cost"] / 10 for p in bootstrap["elements"]}

feature_cols = ["total_points", "minutes", "goals_scored", "assists", "saves", "bonus", "clean_sheets", "yellow_cards", "red_cards", "goals_conceded", "penalties_saved", "penalties_missed", "element_type", "price"]
df = df.dropna(subset=feature_cols)
X = df[feature_cols]

df["predicted_points"] = model.predict(X)
df = df[df["season"] == "2026-27"]
df = df.loc[df.groupby("element")["round"].idxmax()]

# Replace numeric team IDs and prices with current bootstrap values
df["team"] = pd.to_numeric(df["team"], errors="coerce").map(team_map).fillna(df["team"])
df["price"] = df["element"].map(price_map).fillna(df["price"])

df = df.sort_values(by="predicted_points", ascending=False)
print(df[["first_name", "second_name", "team", "predicted_points"]].head(20))
df.to_csv("data/processed/predictions.csv", index=False)
