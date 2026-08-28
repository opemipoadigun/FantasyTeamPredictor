import pandas as pd
from pulp import *

df = pd.read_csv("data/processed/predictions.csv")

price = dict(zip(df.index, df["price"]))
position = dict(zip(df.index, df["element_type"]))
team = dict(zip(df.index, df["team"]))
points = dict(zip(df.index, df["predicted_points"]))

prob = LpProblem("FPL_Team_Selector", LpMaximize)

players = list(df.index)
x = LpVariable.dicts("squad", players, cat="Binary")     # 1 = in squad
s = LpVariable.dicts("start", players, cat="Binary")     # 1 = in starting 11
c = LpVariable.dicts("captain", players, cat="Binary")   # 1 = captain
vc = LpVariable.dicts("vice", players, cat="Binary")     # 1 = vice captain

# Objective: starting 11 points + captain bonus (doubles captain's points)
# VC only activates if captain doesn't play — modelled as a small expected bonus
prob += (
    lpSum(points[i] * s[i] for i in players)
    + lpSum(points[i] * c[i] for i in players)           # captain extra
    + lpSum(points[i] * vc[i] * 0.1 for i in players)   # ~10% chance VC is needed
)

# Squad constraints
prob += lpSum(price[i] * x[i] for i in players) <= 100
prob += lpSum(x[i] for i in players) == 15
prob += lpSum(x[i] for i in players if position[i] == 1) == 2
prob += lpSum(x[i] for i in players if position[i] == 2) == 5
prob += lpSum(x[i] for i in players if position[i] == 3) == 5
prob += lpSum(x[i] for i in players if position[i] == 4) == 3
for t in set(team.values()):
    prob += lpSum(x[i] for i in players if team[i] == t) <= 3

# Starting 11 constraints
for i in players:
    prob += s[i] <= x[i]
prob += lpSum(s[i] for i in players) == 11
prob += lpSum(s[i] for i in players if position[i] == 1) == 1
prob += lpSum(s[i] for i in players if position[i] == 2) >= 3
prob += lpSum(s[i] for i in players if position[i] == 3) >= 2
prob += lpSum(s[i] for i in players if position[i] == 4) >= 1

# Captain and vice captain constraints
prob += lpSum(c[i] for i in players) == 1
prob += lpSum(vc[i] for i in players) == 1
for i in players:
    prob += c[i] <= s[i]           # captain must be in starting 11
    prob += vc[i] <= s[i]          # vice captain must be in starting 11
    prob += c[i] + vc[i] <= 1     # can't be both captain and vice captain

prob.solve(PULP_CBC_CMD(msg=0))

df["in_squad"] = [value(x[i]) for i in players]
df["starts"] = [value(s[i]) for i in players]
df["is_captain"] = [value(c[i]) for i in players]
df["is_vice"] = [value(vc[i]) for i in players]

squad = df[df["in_squad"] == 1][["first_name", "second_name", "team", "element_type", "price", "predicted_points", "starts", "is_captain", "is_vice"]].copy()

def assign_role(row):
    if row["is_captain"] == 1.0:
        return "Captain"
    if row["is_vice"] == 1.0:
        return "Vice Captain"
    if row["starts"] == 1.0:
        return "Starting"
    return "Bench"

squad["role"] = squad.apply(assign_role, axis=1)
squad = squad.sort_values(["starts", "element_type"], ascending=[False, True])
print(squad[["first_name", "second_name", "team", "element_type", "price", "predicted_points", "role"]].to_string(index=False))
squad.to_csv("data/processed/squad.csv", index=False)
