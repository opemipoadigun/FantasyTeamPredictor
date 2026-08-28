import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

FantasyAPI = "https://fantasy.premierleague.com/api/bootstrap-static/"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(FantasyAPI, headers=headers)

if response.status_code == 200:
    data = response.json()
    players = data["elements"]
    print(len(players))
    os.makedirs("data/raw", exist_ok=True)
    with open('data/raw/bootstrap.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
else:
    print("Error fetching bootstrap")


def fetch_player(player):
    url = "https://fantasy.premierleague.com/api/element-summary/{id}/".format(id=player["id"])
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data["history"], data["history_past"]
    except requests.exceptions.RequestException:
        pass
    return [], []

all_history = []
all_history_past = []

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(fetch_player, player): player for player in players}
    for future in as_completed(futures):
        history, history_past = future.result()
        all_history.extend(history)
        all_history_past.extend(history_past)

with open('data/raw/gw_history.json', 'w', encoding='utf-8') as f:
    json.dump(all_history, f, ensure_ascii=False, indent=4)
with open('data/raw/past_history.json', 'w', encoding='utf-8') as f:
    json.dump(all_history_past, f, ensure_ascii=False, indent=4)

Vaastav202526 = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv"
response = requests.get(Vaastav202526)
if response.status_code == 200:
    with open('data/raw/lastseason.csv', 'w') as f:
        f.write(response.text)
else:
    print("Error fetching vaastav data")

fixtures_response = requests.get("https://fantasy.premierleague.com/api/fixtures/", headers=headers)
if fixtures_response.status_code == 200:
    with open('data/raw/fixtures.json', 'w', encoding='utf-8') as f:
        json.dump(fixtures_response.json(), f, ensure_ascii=False, indent=4)
else:
    print("Error fetching fixtures")
