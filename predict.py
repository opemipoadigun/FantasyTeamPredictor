import pandas as pd
import joblib
import json

model = joblib.load("models/model.joblib")
df = pd.read_csv("data/processed/combined_data.csv", low_memory=False)

with open("data/raw/bootstrap.json", "r", encoding="utf-8") as f:
    bootstrap = json.load(f)

team_map = {t["id"]: t["name"] for t in bootstrap["teams"]}
price_map = {p["id"]: p["now_cost"] / 10 for p in bootstrap["elements"]}

# Availability: exclude unavailable players, scale doubtful by chance of playing
unavailable_statuses = {"i", "s", "n", "u"}
availability_map = {}
for p in bootstrap["elements"]:
    if p["status"] in unavailable_statuses:
        availability_map[p["id"]] = 0.0
    elif p["status"] == "d":
        chance = p.get("chance_of_playing_next_round") or 0
        availability_map[p["id"]] = chance / 100
    else:
        availability_map[p["id"]] = 1.0

feature_cols = ["total_points", "minutes", "goals_scored", "assists", "saves", "bonus", "clean_sheets", "yellow_cards", "red_cards", "goals_conceded", "penalties_saved", "penalties_missed", "element_type", "price", "next_fixture_difficulty"]
df = df.dropna(subset=feature_cols)
X = df[feature_cols]

df["predicted_points"] = model.predict(X)
df = df[df["season"] == "2026-27"]
df = df.loc[df.groupby("element")["round"].idxmax()]

# Replace numeric team IDs and prices with current bootstrap values
df["team"] = pd.to_numeric(df["team"], errors="coerce").map(team_map).fillna(df["team"])
df["price"] = df["element"].map(price_map).fillna(df["price"])

# Apply availability: exclude injured/suspended, scale doubtful by chance of playing
df["availability"] = df["element"].map(availability_map).fillna(1.0)
df["predicted_points"] = df["predicted_points"] * df["availability"]
df = df[df["availability"] > 0]  # drop fully unavailable players

df = df.sort_values(by="predicted_points", ascending=False)
print(df[["first_name", "second_name", "team", "predicted_points"]].head(20))
df.to_csv("data/processed/predictions.csv", index=False)
