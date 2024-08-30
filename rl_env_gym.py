
from collections import deque, namedtuple
from typing import Dict, List
import random
import numpy as np
import pandas as pd
import math
import gym
from gym import spaces
# import gymnasium as gym
# from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_SPACE_DIM = 6
STATE_SCHEMA = {"QB_mean": 15, "RB_mean": 30, "WR_mean": 30, "TE_mean": 15, "K_mean": 15, "DEF_mean": 15, 
                "round": 1, "turns_until_next_pick": 1,
                "mgr_0_team": 6, "mgr_1_team": 6, "mgr_2_team": 6,  # these are the team compositions for each manager
                "mgr_3_team": 6, "mgr_4_team": 6, "mgr_5_team": 6,
                "mgr_6_team": 6, "mgr_7_team": 6, "mgr_8_team": 6, 
                "mgr_9_team": 6, "mgr_10_team": 6, "mgr_11_team": 6
                }

STARTER_COMPOSITION = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1, "FLEX": 2}
STATE_SPACE_DIM = np.sum(list(STATE_SCHEMA.values()))
NUM_DRAFT_ROUNDS = 15
NUM_MGRS = 12

def softmax(x, temperature=1.0):
    x = np.array(x)
    e_x = np.exp((x - np.max(x))/temperature)
    return e_x / e_x.sum(axis=0)


class DraftEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(DraftEnv, self).__init__()
        
        df_players = self._make_players_df()
        self.action_space = spaces.Discrete(ACTION_SPACE_DIM)
        self.actions = {0: "QB", 1: "RB", 2: "WR", 3: "TE", 4: "K", 5: "DEF"} 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_SPACE_DIM,), dtype=np.float32)
        self.turns = self._make_turns_list()
      
        self.cur_turn = 0  # increments from 0 to NUM_MGRS * NUM_DRAFT_ROUNDS - 1, indexes self.draft to get the current manager
        self.cur_round = 0  # increments from 0 to NUM_DRAFT_ROUNDS - 1
        self.all_players = df_players.dropna(subset=["mean", "std"]).copy()  # all players
        # print(f'Scaling mean points by {self.all_players["mean"].max()}')
        self.all_players["mean"] = self.all_players["mean"] / self.all_players["mean"].max() # scale points
        self.all_players = self.all_players.sort_values(by="mean", ascending=False).reset_index(drop=True)
        
        
        self.open_players = self.all_players.copy()  # players that are still available to be drafted
        self.draft = self.make_empty_draft()
        self.objective = "fp_mean"
        self.update_state()
    
    def make_empty_draft(self):
        draft = pd.DataFrame({  # the draft board
            "round": [turn["round"] for turn in self.turns],
            "mgr": [turn["mgr"] for turn in self.turns],
            "sleeper_id": None,
            "full_name": None,
            "team": None,
            "position": None, # actual position
            "team_pos": None, # labels some players as FLEX
            "fp_mean": None,
            "fp_std": None
        })
        return draft
        
    def trim_or_pad(self, arr, length):
        ''' Trim or pad an array to a specified length '''
        if len(arr) == 0:
            return np.zeros(length)
        arr = arr[:length]
        pad_len = max(0, length - len(arr))
        arr = np.pad(arr, (0, pad_len), 'constant', constant_values=0)
        return arr
    
    def n_turns_until_next_pick(self):
        ''' Returns the number of turns until the next pick for the current manager '''
        mgr = self.draft["mgr"].values[self.cur_turn]
        if self.cur_round is None or self.cur_round >= NUM_DRAFT_ROUNDS - 1:
            return 0
        next_pick = self.draft.loc[(self.draft["mgr"] == mgr) & (self.draft["round"] == self.cur_round + 1)]
        n_turns_until_next_pick = next_pick.index[0] - self.cur_turn
        n_turns_until_next_pick /= 23  # scale between 0 and 1
        return n_turns_until_next_pick
    
    def get_cur_mgr(self):
        return self.draft["mgr"].values[self.cur_turn]
        
    def update_state(self):
        self.open_players = self.open_players.sort_values(by="mean", ascending=False).reset_index(drop=True)
        # remove chosen players
        self.open_players = self.open_players.loc[~self.open_players["sleeper_id"].isin(self.draft["sleeper_id"].values)]
        
        qb_mean = self.open_players.loc[self.open_players["position"] == "QB"]["mean"].values
        qb_mean = self.trim_or_pad(qb_mean, STATE_SCHEMA["QB_mean"])
        
        rb_mean = self.open_players.loc[self.open_players["position"] == "RB"]["mean"].values
        rb_mean = self.trim_or_pad(rb_mean, STATE_SCHEMA["RB_mean"])
        
        wr_mean = self.open_players.loc[self.open_players["position"] == "WR"]["mean"].values
        wr_mean = self.trim_or_pad(wr_mean, STATE_SCHEMA["WR_mean"])
        
        te_mean = self.open_players.loc[self.open_players["position"] == "TE"]["mean"].values
        te_mean = self.trim_or_pad(te_mean, STATE_SCHEMA["TE_mean"])
        
        k_mean = self.open_players.loc[self.open_players["position"] == "K"]["mean"].values
        k_mean = self.trim_or_pad(k_mean, STATE_SCHEMA["K_mean"])
        
        def_mean = self.open_players.loc[self.open_players["position"] == "DEF"]["mean"].values
        def_mean = self.trim_or_pad(def_mean, STATE_SCHEMA["DEF_mean"])
        
        
        # get all manager team compositions (i.e. count of each position)
        team_comps = []
        for mgr_num in range(NUM_MGRS):
            team_comp = self.get_team_comp(mgr_num, flex=False)
            qbs = team_comp.get("QB", 0)
            rbs = team_comp.get("RB", 0)
            wrs = team_comp.get("WR", 0)
            tes = team_comp.get("TE", 0)
            ks = team_comp.get("K", 0)
            defs = team_comp.get("DEF", 0)
            team = [qbs, rbs, wrs, tes, ks, defs]
            team_comps += team
        
        # print(f"cur round: {self.cur_round}, n_turns_until_next_pick: {self.n_turns_until_next_pick()}")
        self.state = np.concatenate([qb_mean, rb_mean, wr_mean, te_mean, k_mean, def_mean,
                                        [self.cur_round, self.n_turns_until_next_pick()],
                                        team_comps])
        
        return self.state
    
    def get_needed_pos_counts(self, mgr_num: int, flex=True) -> Dict[str, int]:
        ''' Get the number of each position needed for a manager to fill their starters '''
        team_comp = self.get_team_comp(mgr_num, flex=flex)
        needed_pos_counts = {pos: STARTER_COMPOSITION[pos] - team_comp[pos] for pos in STARTER_COMPOSITION}
        return needed_pos_counts

    def reasonable_option(self, mgr_num: int=None) -> int:
        ''' Choose a reasonable option based on the team composition of a manager.
        The reasonable option is the highest point player in a position for which the manager
        still needs to fill a starting position. Else, choose highest point player available.
        '''
        if mgr_num is None:
            mgr_num = self.get_cur_mgr()
        team_comp = self.get_team_comp(mgr_num, flex=True)
        needed_pos_counts = self.get_needed_pos_counts(mgr_num) # includes flex
        
        # first get what non-flex positions are needed
        needed_positions = [pos for pos, count in needed_pos_counts.items() if pos != "FLEX" and count > 0]
        rb_te_wr_filled = team_comp.get("RB", 0) >= STARTER_COMPOSITION["RB"] and \
            team_comp.get("WR", 0) >= STARTER_COMPOSITION["WR"] and \
            team_comp.get("TE", 0) >= STARTER_COMPOSITION["TE"]
        
        # if the rb, wr, and te are filled, check flex
        if needed_pos_counts.get("FLEX", 0) > 0 and rb_te_wr_filled:
            # don't worry about flex if rb, wr, te are not filled yet
                needed_positions += ["RB", "WR", "TE"]
                
        # if we need players, choose the highest point player available of any needed position
        if len(needed_positions) != 0:
            needed_players = self.open_players[self.open_players['position'].isin(needed_positions)]
            if not needed_players.empty:
                # if players are available in a needed position, choose the highest point player in a needed position
                chosen_player = needed_players.iloc[0]
            else:
                # if no players are available in a needed position, choose the highest point player available
                chosen_player = self.open_players.iloc[0]
        else: 
            # if we don't need players, choose the highest point player available
            chosen_player = self.open_players.iloc[0]
        action = [num for num, pos in self.actions.items() if pos == chosen_player["position"]][0]
        return action
    
    def reasonable_option_stoch(self, mgr_num: int=None, temperature: float=.1) -> int:
        ''' Choose a random reasonable option based on the team composition of a manager.
        Note this function doesn't exactly work right, because it ultimately chooses a position
        that is needed, but then chooses the highest point player in that position. 
        '''
        if mgr_num is None:
            mgr_num = self.get_cur_mgr()
        team_comp = self.get_team_comp(mgr_num, flex=True)
        needed_pos_counts = self.get_needed_pos_counts(mgr_num) # includes flex
        
        # first get what non-flex positions are needed
        needed_positions = [pos for pos, count in needed_pos_counts.items() if pos != "FLEX" and count > 0]
        rb_te_wr_filled = team_comp.get("RB", 0) >= STARTER_COMPOSITION["RB"] and \
            team_comp.get("WR", 0) >= STARTER_COMPOSITION["WR"] and \
            team_comp.get("TE", 0) >= STARTER_COMPOSITION["TE"]
        
        # if the rb, wr, and te are filled, check flex
        if needed_pos_counts.get("FLEX", 0) > 0 and rb_te_wr_filled:
            # don't worry about flex if rb, wr, te are not filled yet
                needed_positions += ["RB", "WR", "TE"]
                
        # if we need players, choose the highest point player available of any needed position
        if len(needed_positions) != 0:
            needed_players = self.open_players[self.open_players['position'].isin(needed_positions)]
            if not needed_players.empty:
                # if players are available in a needed position, choose the highest point player in a needed position
                # select up to 5 of the top players, with a max of 100 mean less than the top
                options = needed_players.loc[needed_players["mean"] >= needed_players["mean"].max() - 100]
                options = options.iloc[:np.min([5, len(options)])]
                chosen_player = options.sample(1, weights=softmax(options["mean"].values, temperature=temperature)).iloc[0]
                
            else:
                # if no players are available in a needed position, choose the highest point player available
                # chosen_player = self.open_players.iloc[0]
                options = self.open_players.loc[self.open_players["mean"] >= self.open_players["mean"].max() - 100]
                options = options.iloc[:np.min([5, len(options)])]
                chosen_player = options.sample(1, weights=softmax(options["mean"].values, temperature=temperature)).iloc[0]
                
        else: 
            # if we don't need players, choose the highest point player available
            # chosen_player = self.open_players.iloc[0]
            options = self.open_players.loc[self.open_players["mean"] >= self.open_players["mean"].max() - 100]
            options = options.iloc[:np.min([5, len(options)])]
            chosen_player = options.sample(1, weights=softmax(options["mean"].values, temperature=temperature)).iloc[0]
            
        action = [num for num, pos in self.actions.items() if pos == chosen_player["position"]][0]
        return action
        
        
    def get_team_comp(self, mgr_num: int, flex=True) -> Dict[str, int]:
        ''' Get the team composition for a manager.
        If flex is True, then the FLEX position is included in the count'''
        if flex:
            team_comp = self.draft.loc[(self.draft["mgr"] == mgr_num)].groupby("team_pos").size().to_dict()
            team_comp["FLEX"] = team_comp.get("FLEX", 0)
        else:
            team_comp = self.draft.loc[(self.draft["mgr"] == mgr_num)].groupby("position").size().to_dict()
        for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]:
            # fill in missing positions with 0
            team_comp[pos] = team_comp.get(pos, 0)
        return team_comp
    
    def _make_turns_list(self):
        turns = []
        for round_num in range(15):
            if round_num % 2 == 0:  # Even numbered rounds (0, 2, 4, ...)
                for i in range(12):
                    turns.append({"round": round_num, "mgr": i})
            else:  # Odd numbered rounds (1, 3, 5, ...)
                for i in range(11, -1, -1):
                    turns.append({"round": round_num, "mgr": i})
        return turns
        
        
    def reset(self):
        """Resets the environment to an initial state and returns an initial observation."""
        self.draft = self.make_empty_draft()
        self.cur_round = 0
        self.cur_turn = 0
        self.open_players = self.all_players.copy()
        
        
        self.update_state()
        info = {}
        return self.state, info
    
    def is_starters_filled(self, mgr_num: int) -> bool:
        filled = [x >= STARTER_COMPOSITION[pos] for pos, x in self.get_team_comp(mgr_num, flex=True).items()]
        if all(filled):
            return True
        return False
    
    def get_starters(self, mgr_num: int=None) -> pd.DataFrame:
        if mgr_num is None:
            mgr_num = self.get_cur_mgr()
        team = self.draft.loc[(self.draft["mgr"] == mgr_num)].copy()
        starter_ids = []
        for pos in STARTER_COMPOSITION.keys():
            players_pos = team.loc[team["team_pos"] == pos]
            players_pos = players_pos.sort_values(by=self.objective, ascending=False).reset_index(drop=True)
            pos_starter_ids = players_pos.iloc[:STARTER_COMPOSITION[pos]]['sleeper_id'].values
            if len(pos_starter_ids) > 0:
                starter_ids += list(pos_starter_ids)
        return team.loc[team["sleeper_id"].isin(starter_ids)]
        
            
    def get_bench(self, mgr_num: int=None) -> pd.DataFrame:
        if mgr_num is None:
            mgr_num = self.get_cur_mgr()
        team = self.draft.loc[(self.draft["mgr"] == mgr_num)].copy()
        starter_ids = self.get_starters(mgr_num)['sleeper_id'].values
        return team.loc[~team["sleeper_id"].isin(starter_ids)]
    
    def compute_reward(self, mgr_num: int) -> float:
        

        # print(f'mgr_num: {mgr_num}; round: {self.cur_round}')

        team = self.draft.loc[(self.draft["mgr"] == mgr_num)].copy()
        recent_choice = team.loc[team["round"] == self.cur_round]
        # print(f'recent_choice: {recent_choice}')

        team = team.sort_values(by=self.objective, ascending=False).reset_index(drop=True)
        
        starters = self.get_starters(mgr_num)
        bench = self.get_bench(mgr_num)
        
        starters_sum = starters[self.objective].sum()
        bench_sum = bench[self.objective].sum()
        
        starter_ids = starters["sleeper_id"].values
        recent_choice_id = recent_choice["sleeper_id"].values[0]
        chose_starter = recent_choice_id in starter_ids 
        
        # if self.is_starters_filled(mgr_num):
        #     reward = starters_sum + bench_sum
        #     # reward = starters_sum
            
        # -- rewards if the team is not full -- #
        # give a small reward if they draft a starter
        if self.cur_round < (NUM_DRAFT_ROUNDS - 1) and chose_starter:
            reward = recent_choice[self.objective].values[0]
        else:
            reward = 0
        
        # -- rewards if the team is full -- #
        if self.cur_round == NUM_DRAFT_ROUNDS-1 and not self.is_starters_filled(mgr_num):
        # Penalize if the starters are not filled by the end of the draft
            reward = -1 * NUM_DRAFT_ROUNDS
        elif self.cur_round == NUM_DRAFT_ROUNDS-1 and self.is_starters_filled(mgr_num):
        # give a big reward if starters are full
            reward = starters_sum + .1*bench_sum
        
        reward /= NUM_DRAFT_ROUNDS
        return reward
    
             
    def step(self, action):
        """Run one timestep of the environment's dynamics."""
        
        mgr_num = self.draft["mgr"].values[self.cur_turn]
        
        done = True if self.cur_round == NUM_DRAFT_ROUNDS - 1 else False
        
        # -- Choose the player -- #
        filtered_players = self.open_players.loc[self.open_players["position"] == self.actions[action]]
        filtered_players = filtered_players.sort_values(by="mean", ascending=False).reset_index(drop=True) 
        if not filtered_players.empty:
            chosen_player = filtered_players.iloc[0]
        else:
            # If no players are available for the chosen position, they get no player
            chosen_player = {"mean": 0, "std": 0, 
                             "sleeper_id": None, "full_name": None, 
                             "team": None, "position": None}  # or handle the case where no player is found

        team_comp = self.get_team_comp(mgr_num, flex=False)
        # Indicates if the team is ready to draft a flex player, ie. rb, wr, te are filled
        flex_ready = team_comp.get("RB") >= STARTER_COMPOSITION["RB"] and \
            team_comp.get("WR") >= STARTER_COMPOSITION["WR"] and \
            team_comp.get("TE") >= STARTER_COMPOSITION["TE"]
        # If mgr is ready for flex player, then assign FLEX
        if chosen_player["position"] in ["RB", "WR", "TE"] and flex_ready:
            team_pos = "FLEX"
        else: 
            team_pos = chosen_player["position"]

        # -- Add the player to the draft -- #
        self.draft.loc[self.cur_turn, "sleeper_id"] = chosen_player["sleeper_id"]
        self.draft.loc[self.cur_turn, "full_name"] = chosen_player["full_name"]
        self.draft.loc[self.cur_turn, "team"] = chosen_player["team"]
        self.draft.loc[self.cur_turn, "position"] = chosen_player["position"]
        self.draft.loc[self.cur_turn, "team_pos"] = team_pos
        self.draft.loc[self.cur_turn, "fp_mean"] = chosen_player["mean"]
        self.draft.loc[self.cur_turn, "fp_std"] = chosen_player["std"]


        info = {
            "cur_turn": self.cur_turn,
            "cur_round": self.cur_round,
            "chosen_player": chosen_player,
            "mgr_num": mgr_num
        }  
        
        # update the state
        self.update_state()
        reward = self.compute_reward(mgr_num)
        
        self.cur_turn += 1
        if self.cur_turn < len(self.turns):
            self.cur_round = self.draft["round"].values[self.cur_turn]
        else:
            # even though this round doesn't exist,
            # we need to have a valid int for the state
            self.cur_round = NUM_DRAFT_ROUNDS
        
        # if done:
        #     reward = self.compute_reward(mgr_num)
        #     return self.state, reward, done, info
        # else:
        #     return self.state, 0, done, info
        return self.state, reward, done, info
    

    def get_mgr_draft(self, mgr_num: int) -> pd.DataFrame:
        return self.draft.loc[self.draft["mgr"] == mgr_num]
    
    def get_state(self):
        return self.state
    
    def _make_players_df(self):
        df_sleeper = pd.read_csv("data/sleeper/all_players.csv")
        df_qb_proj = pd.read_csv("data/projections/QB_projections.csv")
        df_rb_proj = pd.read_csv("data/projections/RB_projections.csv")
        df_wr_proj = pd.read_csv("data/projections/WR_projections.csv")
        df_te_proj = pd.read_csv("data/projections/TE_projections.csv")
        df_k_proj = pd.read_csv("data/projections/K_projections.csv")
        df_def_proj = pd.read_csv("data/projections/DEF_projections.csv")



        df_qb_proj = df_qb_proj.loc[:, ["sleeper_id", "full_name", "team", "position", "source", "fpts"]].sort_values(by="fpts", ascending=False)
        df_rb_proj = df_rb_proj.loc[:, ["sleeper_id", "full_name", "team", "position", "source", "fpts"]].sort_values(by="fpts", ascending=False)
        df_wr_proj = df_wr_proj.loc[:, ["sleeper_id", "full_name", "team", "position", "source", "fpts"]].sort_values(by="fpts", ascending=False)
        df_te_proj = df_te_proj.loc[:, ["sleeper_id", "full_name", "team", "position", "source", "fpts"]].sort_values(by="fpts", ascending=False)
        df_k_proj = df_k_proj.loc[:, ["sleeper_id", "full_name", "team", "position", "source", "fpts"]].sort_values(by="fpts", ascending=False)
        df_def_proj = df_def_proj.loc[:, ["sleeper_id", "full_name", "team", "position", "source", "fpts"]].sort_values(by="fpts", ascending=False)

        df_proj = pd.concat([df_qb_proj, df_rb_proj, df_wr_proj, df_te_proj, df_k_proj, df_def_proj])

        df_proj_agg = df_proj.groupby('sleeper_id')['fpts'].agg(['mean', 'std']).reset_index()
        df_proj_agg['sleeper_id'] = df_proj_agg['sleeper_id'].astype(str)


        df_players = df_proj_agg.merge(df_sleeper.loc[:, ['sleeper_id', 'full_name', 'position', 'team']], 
                                        on='sleeper_id', 
                                        how='left')
        return df_players

    
    def render(self, mode='human'):
        """Render the environment to the screen (optional)."""
        print(f"State: {self.state}")
    
    def close(self):
        """Cleanup the environment before closing (optional)."""
        pass
    

