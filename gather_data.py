import pandas as pd
import requests
import os 
import json
import dotenv
import datetime
from datetime import datetime
import time

from sleeper_api import get_all_players, check_all_players_file

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
SLEEPER_ALL_PLAYERS_URL = config["urls"]["sleeper"]["all_players"]
ALL_PLAYERS_FILE = config["files"]["sleeper"]["all_players"]



if __name__ == "__main__":

    if not check_all_players_file():
        print("Updating all players file")
        all_players = get_all_players()
        with open(ALL_PLAYERS_FILE, "w") as f:
            json.dump(all_players, f, indent=2)
    else: 
        print("All players file is up to date")

    with open(ALL_PLAYERS_FILE, "r") as f:
        data = json.load(f)

    # Extract the keys and merge with the dictionaries
    records = [{"id": key, **value} for key, value in data.items()]

    # Convert to DataFrame
    df = pd.DataFrame(records)

    df.to_csv("../data/sleeper/all_players.csv", index=False)