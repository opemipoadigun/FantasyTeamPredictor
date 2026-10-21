import argparse
import os
import pandas as pd
from pulp import *


parser = argparse.ArgumentParser()
parser.add_argument("--free-transfers", type=int, default=1,
                    help="Free transfers available this GW (default 1)")
parser.add_argument("--max-transfers", type=int, default=None,
                    help="Hard cap on transfers (otherwise optimizer can exceed free allowance with penalty)")
parser.add_argument("--wildcard", action="store_true",
                    help="Use wildcard: no transfer limit and no penalty")
args = parser.parse_args()
free_transfers = max(0, args.free_transfers)
hard_cap = args.max_transfers  # None = soft penalty mode


df = pd.read_csv("data/processed/predictions.csv")

price = dict(zip(df.index, df["price"]))
position = dict(zip(df.index, df["element_type"]))
team = dict(zip(df.index, df["team"]))
points = dict(zip(df.index, df["predicted_points"]))
players = list(df.index)

# Load current squad if it exists to enforce transfer costs
current_squad_elements = set()
squad_path = "data/processed/squad.csv"
if os.path.exists(squad_path):
    current_squad = pd.read_csv(squad_path)
    if "element" in current_squad.columns:
        current_squad_elements = set(current_squad["element"].dropna().astype(int))

if args.wildcard:
    current_squad_elements = set()

in_current = {i: (1 if int(df.loc[i, "element"]) in current_squad_elements else 0)
              for i in df.index}

recognised = sum(in_current.values())
if current_squad_elements:
    print(f"[Transfer check] {recognised} of {len(current_squad_elements)} current squad players recognised in predictions")


def build_and_solve(cap=None):
    prob = LpProblem("FPL_Team_Selector", LpMaximize)
    x = LpVariable.dicts("squad", players, cat="Binary")
    s = LpVariable.dicts("start", players, cat="Binary")
    c = LpVariable.dicts("captain", players, cat="Binary")
    vc = LpVariable.dicts("vice", players, cat="Binary")
    transfer_in = LpVariable.dicts("transfer_in", players, cat="Binary")
    excess = LpVariable("excess_transfers", lowBound=0)

    prob += (
        lpSum(points[i] * s[i] for i in players)
        + lpSum(points[i] * c[i] for i in players)
        + lpSum(points[i] * vc[i] * 0.1 for i in players)
        - 4 * excess
    )

    prob += lpSum(price[i] * x[i] for i in players) <= 100
    prob += lpSum(x[i] for i in players) == 15
    prob += lpSum(x[i] for i in players if position[i] == 1) == 2
    prob += lpSum(x[i] for i in players if position[i] == 2) == 5
    prob += lpSum(x[i] for i in players if position[i] == 3) == 5
    prob += lpSum(x[i] for i in players if position[i] == 4) == 3
    for t in set(team.values()):
        prob += lpSum(x[i] for i in players if team[i] == t) <= 3

    for i in players:
        prob += s[i] <= x[i]
    prob += lpSum(s[i] for i in players) == 11
    prob += lpSum(s[i] for i in players if position[i] == 1) == 1
    prob += lpSum(s[i] for i in players if position[i] == 2) >= 3
    prob += lpSum(s[i] for i in players if position[i] == 3) >= 2
    prob += lpSum(s[i] for i in players if position[i] == 4) >= 1

    prob += lpSum(c[i] for i in players) == 1
    prob += lpSum(vc[i] for i in players) == 1
    for i in players:
        prob += c[i] <= s[i]
        prob += vc[i] <= s[i]
        prob += c[i] + vc[i] <= 1

    if current_squad_elements:
        for i in players:
            prob += transfer_in[i] >= x[i] - in_current[i]
            prob += transfer_in[i] <= x[i]
        if cap is not None:
            prob += lpSum(transfer_in[i] for i in players) <= cap
        prob += excess >= lpSum(transfer_in[i] for i in players) - free_transfers

    prob.solve(PULP_CBC_CMD(msg=0))
    return x, s, c, vc, transfer_in, excess


def extract_result(x, s, c, vc, transfer_in):
    result = df.copy()
    result["in_squad"] = [value(x[i]) for i in players]
    result["starts"] = [value(s[i]) for i in players]
    result["is_captain"] = [value(c[i]) for i in players]
    result["is_vice"] = [value(vc[i]) for i in players]
    result["transfer_in"] = [value(transfer_in[i]) for i in players]
    squad = result[result["in_squad"] == 1][
        ["first_name", "second_name", "team", "element_type", "price",
         "predicted_points", "starts", "is_captain", "is_vice", "transfer_in", "element"]
    ].copy()

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
    n_transfers = int(sum(v for v in result["transfer_in"] if v and v == 1.0)) if current_squad_elements else 0
    penalty = max(0, n_transfers - free_transfers) * 4
    gross_pts = sum(
        points[i] * (value(s[i]) or 0)
        + points[i] * (value(c[i]) or 0)
        + points[i] * (value(vc[i]) or 0) * 0.1
        for i in players
    )
    net_pts = gross_pts - penalty
    return squad, n_transfers, penalty, gross_pts, net_pts


# --- Primary solve (soft penalty or hard cap if --max-transfers given) ---
x, s, c, vc, transfer_in, excess = build_and_solve(cap=hard_cap)
squad, n_transfers, penalty, gross_pts, net_pts = extract_result(x, s, c, vc, transfer_in)

print(squad[["first_name", "second_name", "team", "element_type", "price",
             "predicted_points", "role", "transfer_in"]].to_string(index=False))

if args.wildcard:
    print(f"\nWildcard active — {n_transfers} transfer(s) made, no penalty")
elif current_squad_elements:
    label = f"(capped at {hard_cap})" if hard_cap is not None else "(with penalty)"
    print(f"\nTransfers in: {n_transfers} | Free transfers: {free_transfers} | "
          f"Gross pts: {gross_pts:.1f} | Penalty: -{penalty} | Net: {net_pts:.1f} {label}")

    # --- Show comparison when soft mode makes more transfers than free allowance ---
    if hard_cap is None and n_transfers > free_transfers:
        x2, s2, c2, vc2, ti2, ex2 = build_and_solve(cap=free_transfers)
        _, n2, pen2, gross2, net2 = extract_result(x2, s2, c2, vc2, ti2)
        diff = net_pts - net2
        print(f"\nComparison — capped at {free_transfers} transfer(s):")
        print(f"  Soft (accept penalty): {n_transfers} transfers, -{penalty} pts penalty → net {net_pts:.1f} pts")
        print(f"  Capped (no penalty):   {n2} transfers, no penalty → net {net2:.1f} pts")
        if diff > 0:
            print(f"  → Taking the penalty gains {diff:.1f} pts. Run with --max-transfers {free_transfers} to force the cap.")
        else:
            print(f"  → Capping saves {-diff:.1f} pts vs taking the penalty. Run with --max-transfers {free_transfers} to use this instead.")

    elif hard_cap is not None and n_transfers < hard_cap:
        print(f"  (Only {n_transfers} transfer(s) needed — cap of {hard_cap} not reached)")
else:
    print("\nNo existing squad found — picking initial team (no transfer constraints applied)")

squad.to_csv("data/processed/squad.csv", index=False)
