import requests
import os
import json
from datetime import datetime
import pandas as pd
import yaml
import sys

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

SLEEPER_ALL_PLAYERS_URL = config["urls"]["sleeper"]["all_players"]
ALL_PLAYERS_CSV = config["files"]["sleeper"]["all_players_csv"]
ALL_PLAYERS_JSON = config["files"]["sleeper"]["all_players_json"]


def get_all_players():
    response = requests.get(SLEEPER_ALL_PLAYERS_URL)
    response.raise_for_status()
    return response.json()

def check_all_players_file():
    '''return true if the all players file exists from the current day, false otherwise'''
    if os.path.exists(ALL_PLAYERS_JSON):
        file_time = os.path.getmtime(ALL_PLAYERS_JSON)
        file_date = datetime.fromtimestamp(file_time)
        current_date = datetime.now()
        if file_date.date() == current_date.date():
            return True
    return False


if __name__ == "__main__":

    if check_all_players_file():
        print("All players file is up to date")
    else:
        print("Updating all players file")
        all_players = get_all_players()
        with open(ALL_PLAYERS_JSON, "w") as f:
            json.dump(all_players, f, indent=2)
                
    with open(ALL_PLAYERS_JSON, "r") as f:
        data = json.load(f)

    # Extract the keys and merge with the dictionaries
    records = [{"id": key, **value} for key, value in data.items()]

    # Convert to DataFrame
    df_players = pd.DataFrame(records)
    
    # get and save defense
    df_defense = df_players[df_players["position"] == "DEF"]
    
    # read adp file
    df_adp = pd.read_csv(config["files"]["sleeper"]["adp"])
    
    df_adp["Player Id"] = df_adp["Player Id"].astype(str)
    
    # get integer rank in positional_rank column
    df_adp["rank_int"] = df_adp["Positional Rank"].str.extract("(\d+)").astype(int)
    
    # filter to only rank_int <= 20
    df_adp = df_adp[df_adp["rank_int"] <= 20]
    
    # filter df_players such that id is in df_adp["Player ID"]
    df_players = df_players[df_players["id"].isin(df_adp["Player Id"].astype(str))]
    
    # filter df_players to relevant positions
    positions = ["QB", "RB", "WR", "TE", "K"]
    df_players = df_players[df_players["position"].isin(positions)]
    
    # add Positional Rank to df_players and keep only the columns Positional Rank and Redraft Half PPR ADP
    df_players = df_players.merge(df_adp[["Player Id", "Positional Rank", "Redraft Half PPR ADP", "Date"]], left_on="id", right_on="Player Id", how="left")
    df_players.rename(columns={"Positional Rank": "positional_rank", "Redraft Half PPR ADP": "adp", "Date": "adp_date"}, inplace=True)
    
    # Add defense to data frame
    df_players = pd.concat([df_players, df_defense])

    df_players.to_csv(ALL_PLAYERS_CSV, index=False)