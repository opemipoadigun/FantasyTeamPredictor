# 1. Import the libraries you need
#    - one for making HTTP requests to the API
#    - one for working with JSON data (hint: it's built into Python)
import time
import os
import requests
import json

# 2. Store the API URL in a variable
#    https://fantasy.premierleague.com/api/bootstrap-static/

FantasyAPI = "https://fantasy.premierleague.com/api/bootstrap-static/"

# 3. Make a GET request to that URL and store the response

response = requests.get(FantasyAPI)

# 4. Parse the response as JSON
if response.status_code == 200:
    data = response.json()



    # 5. The data has a key called "elements" — that's the list of all players
    #    Pull that out into its own variable
    players = data["elements"]

    # 6. Print how many players there are (sanity check)
    print(len(players))

    # 7. Save the raw JSON response to a file called data/raw/bootstrap.json
    #    (create the data/raw/ folder first if it doesn't exist)
    os.makedirs("data/raw", exist_ok=True)
    with open('data/raw/bootstrap.json', 'w', encoding ='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    all_history = []
    all_history_past =[]
    for player in players:
        
        print(f"Fetching player {player['id']}...")
        FantasyPlayerAPI = "https://fantasy.premierleague.com/api/element-summary/{id}/".format(id=player["id"])

        try:
            response = requests.get(FantasyPlayerAPI)
            time.sleep(0.5)
            if response.status_code == 200:
                data = response.json()

                all_history.extend(data["history"])
                all_history_past.extend(data["history_past"])
        except requests.exceptions.RequestException:
            continue


    with open('data/raw/gw_history.json', 'w', encoding ='utf-8') as f: 
        json.dump(all_history, f, ensure_ascii=False, indent=4)
    with open('data/raw/past_history.json', 'w', encoding ='utf-8') as f:
        json.dump(all_history_past, f, ensure_ascii=False, indent=4)

else:
    print("Error")

