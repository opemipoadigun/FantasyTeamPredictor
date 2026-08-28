import pandas as pd
import json

with open('data/raw/bootstrap.json', 'r', encoding='utf-8') as f:
    bootstrap = json.load(f)
with open('data/raw/gw_history.json', 'r', encoding='utf-8') as f:
    history = json.load(f)

bootstrap_df = pd.DataFrame(bootstrap["elements"])[["id", "first_name", "second_name"]]
bootstrap_df["full_name"] = bootstrap_df["first_name"] + " " + bootstrap_df["second_name"]

gw1_df = pd.DataFrame(history)
gw1_df = gw1_df[gw1_df["round"] == 1][["element", "total_points"]]

gw1_df = gw1_df.merge(bootstrap_df, left_on="element", right_on="id")

predictions_df = pd.read_csv("data/processed/predictions.csv")[["first_name", "predicted_points"]]

comparison = gw1_df.merge(predictions_df, left_on="full_name", right_on="first_name", how="inner")
comparison = comparison[["full_name", "predicted_points", "total_points"]]
comparison["error"] = (comparison["predicted_points"] - comparison["total_points"]).abs()
comparison = comparison.sort_values("predicted_points", ascending=False)

print(comparison.to_string(index=False))
print(f"\nMatched players: {len(comparison)}")
print(f"Mean Absolute Error: {comparison['error'].mean():.3f}")
