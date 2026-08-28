import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

model = joblib.load("models/model.joblib")

#Load data/processed/players.csv
df = pd.read_csv("data/processed/combined_data.csv")

feature_cols = ["total_points", "minutes", "goals_scored", "assists", "saves", "bonus", "clean_sheets", "yellow_cards", "red_cards", "goals_conceded", "penalties_saved", "penalties_missed", "element_type", "price"]
df = df.dropna(subset=feature_cols)
X = df[feature_cols]

df["predicted_points"] = model.predict(X)
gw_mae=df.dropna(subset=["next_gw_points"]).groupby("round").apply(lambda x: mean_absolute_error(x["next_gw_points"], x["predicted_points"]))

print("Mean Absolute Error for each Gameweek:")
print(gw_mae)