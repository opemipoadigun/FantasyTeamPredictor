import json

with open("fpl_data.json", "r") as f:
    data = json.load(f)

import pandas as pd

players = data['elements'] 

df = pd.DataFrame(players)

# Pick only columns we care about
columns_of_interest = ["first_name", "second_name", "team", "minutes", "goals_scored", "assists", "saves", "yellow_cards", "red_cards"]
print(df[columns_of_interest].head())

goalkeepers = df[df['element_type'] == 1]
print(goalkeepers[["first_name", "second_name", "saves", "minutes"]].head())

goalkeepers['saves_3plus'] = goalkeepers['saves'] >= 3
print(goalkeepers[['first_name', 'second_name', 'saves', 'saves_3plus']].head())

X = goalkeepers[['minutes']]
y = goalkeepers['saves_3plus']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Check accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))