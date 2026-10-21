# 1. Import libraries (pandas, and from sklearn: train_test_split, 
#    and xgboost: XGBRegressor)
import os
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# 2. Load data/processed/players.csv
df = pd.read_csv("data/processed/combined_data.csv", low_memory=False)

# 3. Define your features (X) — all columns except name, team, and total_points
#    Define your target (y) — total_points column only
feature_cols = ["total_points", "minutes", "goals_scored", "assists", "saves", "bonus", "clean_sheets", "yellow_cards", "red_cards", "goals_conceded", "penalties_saved", "penalties_missed", "element_type", "price", "next_fixture_difficulty"]
df = df.dropna(subset=feature_cols + ["next_gw_points"])
X = df[feature_cols]
y = df['next_gw_points']
# 4. Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Weight current season rows 3x higher so recent form drives predictions more
season_train = df.loc[X_train.index, "season"]
weights = season_train.map(lambda s: 3.0 if s == "2026-27" else 1.0)

# 5. Create the XGBoost model
model = XGBRegressor()
# 6. Train it on the training set
model.fit(X_train, y_train, sample_weight=weights)
# 7. Test it on the test set and print the score
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")