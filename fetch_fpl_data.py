 
import requests
import json

url= "https://fantasy.premierleague.com/api/bootstrap-static/"
response = requests.get(url)
data = response.json()

with open("fpl_data.json", "w") as f:
    json.dump(data, f, indent=2)