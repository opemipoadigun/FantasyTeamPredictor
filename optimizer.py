# 1. Import libraries (pandas, pulp)
import pandas as pd
from pulp import *

# 2. Load the predictions from predict.py output
#    (you'll need to save predictions to a CSV first — add that to predict.py)
df = pd.read_csv("data/processed/predictions.csv")
# 3. Set up the constraints (budget, position counts, max per team)
price = dict(zip(df.index, df["price"]))
position = dict(zip(df.index, df["element_type"]))
team = dict(zip(df.index, df["team"]))
points = dict(zip(df.index, df["predicted_points"]))

# 4. Create a PuLP problem and decision variables
#    (one variable per player — 1 if selected, 0 if not)
prob = LpProblem("FPL_Team_Selector", LpMaximize)

players = list(df.index)
x = LpVariable.dicts("player", players, cat="Binary")

# 5. Add the objective — maximise total predicted points
prob += lpSum(points[i] * x[i] for i in players)

# 6. Add the constraints
prob += lpSum(price[i] * x[i] for i in players) <= 100  # budget
prob += lpSum(x[i] for i in players) == 15              # squad size
prob += lpSum(x[i] for i in players if position[i] == 1) == 2  # GK
prob += lpSum(x[i] for i in players if position[i] == 2) == 5  # DEF
prob += lpSum(x[i] for i in players if position[i] == 3) == 5  # MID
prob += lpSum(x[i] for i in players if position[i] == 4) == 3  # FWD
for t in set(team.values()):
    prob += lpSum(x[i] for i in players if team[i] == t) <= 3  # max 3 per team
# 7. Solve it
prob.solve()
# 8. Print the selected squad
df["selected"] = [value(x[i]) for i in players]
squad = df[df["selected"] == 1][["first_name", "second_name", "team", "element_type", "price", "predicted_points"]]
print(squad)