# 1. Import libraries (pandas, joblib)
import pandas as pd
import joblib

# 2. Load the saved model from models/model.joblib
model = joblib.load("models/model.joblib")
# 3. Load the processed players data from data/processed/players.csv
df = pd.read_csv("data/processed/players.csv")

# 4. Prepare the features — same columns you trained on
#    (drop first_name, second_name, team, total_points)
#    Remember to convert the string columns to numeric again
df["form"] = pd.to_numeric(df["form"], errors="coerce")
df["ict_index"] = pd.to_numeric(df["ict_index"], errors="coerce")
df["expected_goals"] = pd.to_numeric(df["expected_goals"], errors="coerce")
df["expected_assists"] = pd.to_numeric(df["expected_assists"], errors="coerce")
df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce")
df["points_per_game"] = pd.to_numeric(df["points_per_game"], errors="coerce")
X = df.drop(columns =['first_name','second_name','team','total_points'])

# 5. Run the model on ALL players (not just test set — we want predictions for everyone)

# 6. Add the predictions as a new column on the dataframe called "predicted_points"
df["predicted_points"] = model.predict(X)

# 7. Sort players by predicted_points descending so best players are at the top
df = df.sort_values(by='predicted_points', ascending=False)

# 8. Print the top 20 players and their predicted points
print(df[["first_name", "second_name","predicted_points"]].head(20))
df.to_csv("data/processed/predictions.csv",index =False)