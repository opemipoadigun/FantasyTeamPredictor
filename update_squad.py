import argparse
import os
import pandas as pd
import sys

parser = argparse.ArgumentParser(description="View or modify your current FPL squad before running the optimizer")
parser.add_argument("--remove", type=str, help="Player name to remove (partial match)")
parser.add_argument("--add", type=str, help="Player name to add from predictions (partial match)")
parser.add_argument("--reset", action="store_true", help="Clear the saved squad entirely (next optimizer run picks freely)")
parser.add_argument("--build", action="store_true", help="Interactively set your full 15-player squad from scratch")
args = parser.parse_args()

squad_path = "data/processed/squad.csv"
predictions_path = "data/processed/predictions.csv"


def lookup_player(name, predictions, bootstrap_elements=None, team_map=None):
    predictions["full_name"] = (predictions["first_name"] + " " + predictions["second_name"].fillna("")).str.strip()
    mask = predictions["full_name"].str.contains(name, case=False, na=False)
    matches = predictions[mask]
    if matches.empty and bootstrap_elements is not None:
        # Fall back to bootstrap for injured/unavailable players
        bootstrap_matches = [
            p for p in bootstrap_elements
            if name.lower() in (p["first_name"] + " " + p["second_name"]).lower()
        ]
        if not bootstrap_matches:
            print(f"  No player found matching '{name}'")
            return None
        if len(bootstrap_matches) > 1:
            print(f"  Multiple matches in bootstrap — pick one:")
            for i, p in enumerate(bootstrap_matches):
                team = team_map.get(p["team"], p["team"]) if team_map else p["team"]
                print(f"    {i}: {p['first_name']} {p['second_name']} ({team}, £{p['now_cost']/10}m) [{p['status']}]")
            choice = input("  Enter index: ").strip()
            try:
                p = bootstrap_matches[int(choice)]
            except (IndexError, ValueError):
                print("  Invalid choice.")
                return None
        else:
            p = bootstrap_matches[0]
        team = team_map.get(p["team"], p["team"]) if team_map else p["team"]
        print(f"  Found in bootstrap (status: {p['status']}): {p['first_name']} {p['second_name']} ({team})")
        return pd.Series({
            "first_name": p["first_name"],
            "second_name": p["second_name"],
            "element": p["id"],
            "element_type": p["element_type"],
            "team": team,
            "price": p["now_cost"] / 10,
            "predicted_points": 0.0,
        })
    if len(matches) > 1:
        print(f"  Multiple matches — pick one:")
        for idx, row in matches.iterrows():
            print(f"    {idx}: {row['full_name']} ({row['team']}, {['GK','DEF','MID','FWD'][int(row['element_type'])-1]}, £{row['price']}m)")
        choice = input("  Enter index: ").strip()
        try:
            return matches.loc[int(choice)]
        except (KeyError, ValueError):
            print("  Invalid choice.")
            return None
    return matches.iloc[0]


if args.reset:
    if os.path.exists(squad_path):
        os.remove(squad_path)
        print("Squad cleared. Next optimizer run will pick freely with no transfer constraints.")
    else:
        print("No saved squad found.")
    sys.exit(0)

if args.build:
    if not os.path.exists(predictions_path):
        print("No predictions file found. Run predict.py first.")
        sys.exit(1)
    predictions = pd.read_csv(predictions_path)
    with open("data/raw/bootstrap.json", "r", encoding="utf-8") as f:
        import json
        bootstrap = json.load(f)
    bootstrap_elements = bootstrap["elements"]
    team_map = {t["id"]: t["name"] for t in bootstrap["teams"]}
    print("\nEnter your 15 players (2 GK, 5 DEF, 5 MID, 3 FWD).")
    print("Type a name (partial match works) and press Enter. Type 'undo' to redo the last entry.\n")

    slots = [
        ("GK 1", 1), ("GK 2", 1),
        ("DEF 1", 2), ("DEF 2", 2), ("DEF 3", 2), ("DEF 4", 2), ("DEF 5", 2),
        ("MID 1", 3), ("MID 2", 3), ("MID 3", 3), ("MID 4", 3), ("MID 5", 3),
        ("FWD 1", 4), ("FWD 2", 4), ("FWD 3", 4),
    ]

    selected = []
    i = 0
    while i < len(slots):
        slot_name, pos = slots[i]
        name = input(f"  {slot_name}: ").strip()
        if name.lower() == "undo":
            if selected:
                removed = selected.pop()
                print(f"  Undone: {removed['full_name']}")
                i -= 1
            else:
                print("  Nothing to undo.")
            continue
        player = lookup_player(name, predictions.copy(), bootstrap_elements, team_map)
        if player is None:
            continue
        player = player.copy()
        player["starts"] = 0.0
        player["is_captain"] = 0.0
        player["is_vice"] = 0.0
        player["transfer_in"] = 0.0
        player["role"] = "Bench"
        player["full_name"] = (str(player["first_name"]) + " " + str(player.get("second_name", ""))).strip()
        print(f"  Added: {player['full_name']} ({player['team']}, £{player['price']}m)")
        selected.append(player)
        i += 1

    squad = pd.DataFrame(selected).drop(columns=["full_name"], errors="ignore")
    squad.to_csv(squad_path, index=False)
    print("\nSquad saved:")
    print(squad[["first_name", "second_name", "team", "element_type", "price"]].to_string(index=False))
    print("\nRun optimizer.py to get recommendations based on this squad.")
    sys.exit(0)

if not os.path.exists(squad_path):
    print("No saved squad found. Run optimizer.py or use --build to set your squad.")
    sys.exit(1)

squad = pd.read_csv(squad_path)
squad["full_name"] = (squad["first_name"] + " " + squad["second_name"].fillna("")).str.strip()

if not args.remove and not args.add:
    print("\nCurrent squad:")
    print(squad[["first_name", "second_name", "team", "element_type", "price", "role"]].to_string(index=False))
    sys.exit(0)

if args.remove:
    mask = squad["full_name"].str.contains(args.remove, case=False, na=False)
    matches = squad[mask]
    if matches.empty:
        print(f"No player found matching '{args.remove}'")
        sys.exit(1)
    if len(matches) > 1:
        print(f"Multiple matches — be more specific:")
        print(matches[["first_name", "second_name", "team"]].to_string(index=False))
        sys.exit(1)
    print(f"Removing: {matches.iloc[0]['full_name']} ({matches.iloc[0]['team']})")
    squad = squad[~mask]

if args.add:
    if not os.path.exists(predictions_path):
        print("No predictions file found. Run predict.py first.")
        sys.exit(1)
    predictions = pd.read_csv(predictions_path)
    with open("data/raw/bootstrap.json", "r", encoding="utf-8") as f:
        import json
        bootstrap = json.load(f)
    bootstrap_elements = bootstrap["elements"]
    team_map_add = {t["id"]: t["name"] for t in bootstrap["teams"]}
    player = lookup_player(args.add, predictions, bootstrap_elements, team_map_add)
    if player is None:
        sys.exit(1)
    player = player.copy()
    player["starts"] = 0.0
    player["is_captain"] = 0.0
    player["is_vice"] = 0.0
    player["transfer_in"] = 0.0
    player["role"] = "Bench"
    print(f"Adding: {player['first_name']} {player.get('second_name', '')} ({player['team']}, £{player['price']}m)")
    squad = pd.concat([squad, player.to_frame().T], ignore_index=True)

squad = squad.drop(columns=["full_name"], errors="ignore")
squad.to_csv(squad_path, index=False)

print("\nUpdated squad:")
print(squad[["first_name", "second_name", "team", "element_type", "price", "role"]].to_string(index=False))
print("\nRun optimizer.py to get new recommendations based on this squad.")
