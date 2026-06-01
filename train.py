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
df = pd.read_csv("data/processed/players.csv")
df["form"] = pd.to_numeric(df["form"], errors="coerce")
df["ict_index"] = pd.to_numeric(df["ict_index"], errors="coerce")
df["expected_goals"] = pd.to_numeric(df["expected_goals"], errors="coerce")
df["expected_assists"] = pd.to_numeric(df["expected_assists"], errors="coerce")
df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce")
df["points_per_game"] = pd.to_numeric(df["points_per_game"], errors="coerce")

# 3. Define your features (X) — all columns except name, team, and total_points
#    Define your target (y) — total_points column only
X = df.drop(columns =['first_name','second_name','team','total_points'])
y = df['total_points']
# 4. Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
# 5. Create the XGBoost model
model = XGBRegressor()
# 6. Train it on the training set
model.fit(X_train, y_train)
# 7. Test it on the test set and print the score
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")