
import pandas as pd

seasons = ["2024_PRE", "2023_POST", "2023_REG", "2023_PRE", "2022_POST", "2022_REG", "2022_PRE", "2021_POST", "2021_REG", "2021_PRE", "2020_POST", "2020_REG", "2019_POST", 
           "2019_REG", "2018_POST", "2018_REG", "2017_POST", "2017_REG", "2016_POST", "2016_REG", "2015_POST", "2015_REG", "2014_POST", "2014_REG", "2013_POST", "2013_REG",
              "2012_POST", "2012_REG", "2011_POST", "2011_REG", "2010_POST", "2010_REG", "2009_POST", "2009_REG", "2008_POST", "2008_REG", "2007_POST", "2007_REG", "2006_POST",
                "2006_REG", "2005_POST", "2005_REG", "2004_POST", "2004_REG", "2003_POST", "2003_REG", "2002_POST", "2002_REG", "2001_POST", "2001_REG"]

cols = {
    "RK": "rank", "NAME": "name", "TEAM": "team", "POS": "position", "GP": "games_played", "FPTS": "fpts", "FPTS/G": "fpts_per_game", 
    "YDS": "pass_yds", "TD": "pass_td", "INT": "pass_int", "YDS.1": "rush_yds", "TD.1": "rush_td", "REC": "rec", "YDS.2": "rec_yds",
    "TD.2": "rec_td", "SCK": "sacks", "INT.1": "def_int", "FF": "fumbles_forced", "FR": "fumbles_rec"
}

dfs = []
for season in seasons:
    
    url = f"https://fantasydata.com/nfl/fantasy-football-leaders?scope=season&scoring=fpts_half_ppr&order_by=fpts_half_ppr&sort_dir=desc&sp={season}"    
    tables = pd.read_html(url, header=1)
    df = tables[0]
    df.rename(columns=cols, inplace=True)
    df["season"] = season
    df["year"] = season.split("_")[0]
    df["ssn_type"] = season.split("_")[-1]
    dfs.append(df)
    # df.to_csv(f"data/fantasydata/fpts_ssn_{season}.csv", index=False)
    
df = pd.concat(dfs)
df.to_csv(config["files"]["fantasydata"]["fpts_all_ssns"], index=False)