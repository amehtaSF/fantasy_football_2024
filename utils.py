import os
import pandas as pd

def load_projections(dir):
    positions = ["QB", "RB", "WR", "TE", "DEF", "K"]
    dfs = {}
    for pos in positions:
        file = pos + "_projections.csv"
        if os.path.exists(os.path.join(dir, file)):
            dfs[pos] = pd.read_csv(os.path.join(dir, file))
    return dfs