import json
import pandas as pd
from collections import defaultdict

with open("data/raw/bootstrap.json", "r", encoding="utf-8") as f:
    bootstrap = json.load(f)
with open("data/raw/fixtures.json", "r", encoding="utf-8") as f:
    fixtures = json.load(f)

squad = pd.read_csv("data/processed/squad.csv")

# Current gameweek
current_gw = next((e["id"] for e in bootstrap["events"] if e["is_current"]), None)
if current_gw is None:
    current_gw = next((e["id"] for e in bootstrap["events"] if e["is_next"]), 1) - 1

all_team_ids = set(t["id"] for t in bootstrap["teams"])
team_name_map = {t["id"]: t["name"] for t in bootstrap["teams"]}

# Count fixtures per team per GW
team_fixtures_per_gw = defaultdict(lambda: defaultdict(int))
for fixture in fixtures:
    gw = fixture.get("event")
    if gw is None:
        continue
    team_fixtures_per_gw[gw][fixture["team_h"]] += 1
    team_fixtures_per_gw[gw][fixture["team_a"]] += 1

# Squad team IDs (numeric) — recover from bootstrap by name match
team_name_to_id = {t["name"]: t["id"] for t in bootstrap["teams"]}
squad["team_id"] = squad["team"].map(team_name_to_id)

starters = squad[squad["starts"] == 1.0]
bench = squad[squad["starts"] == 0.0]
captain_row = squad[squad["role"] == "Captain"].iloc[0]

look_ahead = 6

print(f"Current GW: {current_gw}")
print(f"Analysing GW{current_gw + 1} to GW{current_gw + look_ahead}\n")

chip_scores = []

for gw in range(current_gw + 1, current_gw + look_ahead + 1):
    gw_fixtures = team_fixtures_per_gw[gw]
    teams_playing = set(gw_fixtures.keys())
    teams_not_playing = all_team_ids - teams_playing
    dgw_teams = {t for t, count in gw_fixtures.items() if count >= 2}

    # How many of my squad have DGW fixtures
    squad_dgw = squad[squad["team_id"].isin(dgw_teams)]
    starters_dgw = starters[starters["team_id"].isin(dgw_teams)]
    bench_dgw = bench[bench["team_id"].isin(dgw_teams)]

    # How many of my starting 11 have BGW (no fixture)
    starters_blank = starters[starters["team_id"].isin(teams_not_playing)]

    # Bench Boost value: extra points from bench players who have DGW
    # (bench players who play twice contribute twice; normal bench contribution already 0)
    bench_boost_extra = bench_dgw["predicted_points"].sum()
    bench_boost_normal = bench["predicted_points"].sum()
    # In a DGW a bench player with 2 fixtures contributes ~2x; bench_boost chip gives bench points
    # Extra vs not using chip = bench total points (they count vs normally 0)
    bench_boost_value = bench["predicted_points"].apply(
        lambda p: p * gw_fixtures.get(squad[squad["predicted_points"] == p]["team_id"].values[0], 1)
        if len(squad[squad["predicted_points"] == p]) > 0 else p
    ).sum()

    # Triple Captain value: captain plays twice in DGW → points tripled across 2 games
    captain_team = captain_row["team_id"]
    captain_fixtures = gw_fixtures.get(captain_team, 1)
    captain_pts = captain_row["predicted_points"]
    # Normal captain: pts * 2 * fixtures (with 2x multiplier)
    # Triple captain: pts * 3 * fixtures
    # Extra from TC vs normal captain: pts * 1 * fixtures
    tc_extra = captain_pts * captain_fixtures

    # Free Hit value: how many of my starters are blanking
    free_hit_value = len(starters_blank)

    chip_scores.append({
        "GW": gw,
        "DGW teams": len(dgw_teams),
        "BGW teams": len(teams_not_playing),
        "My squad DGW": len(squad_dgw),
        "My starters blank": len(starters_blank),
        "Bench Boost extra pts": round(bench["predicted_points"].sum(), 1),
        "TC extra pts": round(tc_extra, 2),
        "Free Hit need": free_hit_value,
    })

scores_df = pd.DataFrame(chip_scores)
print(scores_df.to_string(index=False))

print("\n--- Chip Recommendations ---\n")

for _, row in scores_df.iterrows():
    gw = int(row["GW"])
    suggestions = []

    if row["BGW teams"] >= 6 and row["My starters blank"] >= 3:
        suggestions.append(
            f"FREE HIT — {int(row['BGW teams'])} teams blank, {int(row['My starters blank'])} of your starters have no fixture"
        )

    if row["DGW teams"] >= 8 and row["My squad DGW"] >= 8:
        suggestions.append(
            f"BENCH BOOST — {int(row['DGW teams'])} DGW teams, {int(row['My squad DGW'])} of your squad play twice (+{row['Bench Boost extra pts']} extra bench pts)"
        )
    elif row["DGW teams"] >= 6 and row["My squad DGW"] >= 5:
        suggestions.append(
            f"BENCH BOOST (consider) — {int(row['My squad DGW'])} squad players have DGW"
        )

    if row["TC extra pts"] >= 8 and row["DGW teams"] > 0:
        cap_name = f"{captain_row['first_name']} {captain_row['second_name']}"
        suggestions.append(
            f"TRIPLE CAPTAIN — {cap_name} has DGW, TC worth +{row['TC extra pts']:.1f} extra pts over normal captaincy"
        )
    elif row["TC extra pts"] >= 6:
        cap_name = f"{captain_row['first_name']} {captain_row['second_name']}"
        suggestions.append(
            f"TRIPLE CAPTAIN (consider) — {cap_name} predicted {captain_row['predicted_points']:.1f} pts, TC adds +{row['TC extra pts']:.1f} over normal"
        )

    if suggestions:
        print(f"GW{gw}:")
        for s in suggestions:
            print(f"  * {s}")
    else:
        print(f"GW{gw}: No chip recommended — hold chips for better opportunity")

print(f"\nNote: Wildcard is best used to restructure your squad ahead of a confirmed DGW.")
print(f"Save it until you can see at least 2 GWs ahead clearly on the fixture list.")
