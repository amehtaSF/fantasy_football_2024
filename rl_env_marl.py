
from collections import deque, namedtuple
from typing import Dict, List
import random
import numpy as np
import pandas as pd
import math
# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_SPACE_DIM = 6
STATE_SCHEMA = {"QB_mean": 10, "RB_mean": 15, "WR_mean": 15, "TE_mean": 10, "K_mean": 10, "DEF_mean": 10, 
                "round": 1, "turns_until_next_pick": 1,
                # "cur_team_means": 15, 
                # "cur_team_pos": 15*6, # one hot encoding of position
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

class MARLDraftEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    
    # i'm gonna need to set up a system where there is a dummy round at the end of the draft
    # and then the reward is calculated for that round
    # for that i will need to modify increment_turn, step,
    def __init__(self):
        super(MARLDraftEnv, self).__init__()
        

        df_players = self._make_players_df()
        # self.dummy_round = False # indicates if the round is a dummy round
        self.stochastic_temp = .5
        self.action_space = spaces.Discrete(ACTION_SPACE_DIM)
        self.actions = {0: "QB", 1: "RB", 2: "WR", 3: "TE", 4: "K", 5: "DEF"} 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_SPACE_DIM,), dtype=np.float32)
        self.turns = self._make_turns_list()
      
        self.cur_turn = 0  # increments from 0 to NUM_MGRS * NUM_DRAFT_ROUNDS - 1, indexes self.draft to get the current manager
        self.cur_round = 0  # increments from 0 to NUM_DRAFT_ROUNDS - 1
        self.cur_mgr = 0  # current manager, incremented by increment turn.
        self.all_players = df_players.dropna(subset=["mean", "std"]).copy()  # all players
        # penalize defense bc i don't want them drafted too early
        self.all_players.loc[self.all_players["position"] == "DEF", "mean"] = self.all_players.loc[self.all_players["position"] == "DEF", "mean"] - 15
        self.all_players["mean"] = self.all_players["mean"] / self.all_players["mean"].max() # scale points
        self.all_players = self.all_players.sort_values(by="mean", ascending=False).reset_index(drop=True)
        
        
        # self.open_players = self.all_players.copy()  # players that are still available to be drafted
        self.draft = self._make_empty_draft()
        self.objective = "mean"
        self.bye_weeks = pd.read_csv("data/bye_weeks_2024.csv")
        
        self.keepers = {
            0: {"round": 5, "sleeper_id": "8146"},
            1: {"round": 10, "sleeper_id": "2749"},
            2: {"round": 9, "sleeper_id": "9226"},
            3: {"round": 11, "sleeper_id": "8183"},
            4: {"round": 9, "sleeper_id": "1264"},
            5: {"round": 11, "sleeper_id": "5947"},
            6: {"round": 6, "sleeper_id": "8150"},
            7: {"round": 6, "sleeper_id": "10229"},
            8: {"round": 5, "sleeper_id": "6803"},
            9: {"round": 3, "sleeper_id": "2216"},
            10: {"round": 4, "sleeper_id": "5892"},
            11: {"round": 11, "sleeper_id": "10859"}
        }
        
        self.reset_open_players() # remove keepers
        self.update_state()
        
    def _make_empty_draft(self):
        draft = pd.DataFrame({  # the draft board
            "round": [turn["round"] for turn in self.turns],
            "mgr": [turn["mgr"] for turn in self.turns],
            "sleeper_id": None,
            "full_name": None,
            "team": None,
            "position": None, # actual position
            "team_pos": None, # labels some players as FLEX
            "mean": None,
            "std": None
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
    
    def dummy_position(self, pos):
        ''' Returns a one hot encoding of a position '''
        if pos == "QB":
            return [1, 0, 0, 0, 0, 0]
        if pos == "RB":
            return [0, 1, 0, 0, 0, 0]
        if pos == "WR":
            return [0, 0, 1, 0, 0, 0]
        if pos == "TE":
            return [0, 0, 0, 1, 0, 0]
        if pos == "K":
            return [0, 0, 0, 0, 1, 0]
        if pos == "DEF":
            return [0, 0, 0, 0, 0, 1]
        
    def update_state(self):
        '''
        Update the state of the environment 
        '''

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
        
        # # make a list of mean for each player on current managers team, then pad to 15
        # cur_team_means = self.get_team(mgr_num=self.get_cur_mgr())["mean"].values
        # cur_team_means = self.trim_or_pad(cur_team_means, STATE_SCHEMA["cur_team_means"])
        
        # cur_team_pos = self.get_team(mgr_num=self.get_cur_mgr())["position"].values
        # # convert strings to one hot encoding
        # if all([pos is None for pos in cur_team_pos]):
        #     cur_team_pos = np.zeros(6*NUM_DRAFT_ROUNDS)
        # else: 
        #     cur_team_pos = [self.dummy_position(pos) for pos in cur_team_pos if pos is not None]
        #     print(cur_team_pos)
        #     cur_team_pos = np.concatenate(cur_team_pos)
        #     cur_team_pos = self.trim_or_pad(cur_team_pos, STATE_SCHEMA["cur_team_pos"])
        
        
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
                                        # cur_team_means, cur_team_pos, 
                                        team_comps]).astype(np.float32)
        
        
        return self.state
    
    def get_needed_pos_counts(self, mgr_num: int, flex=True) -> Dict[str, int]:
        ''' Get the number of each position needed for a manager to fill their starters '''
        team_comp = self.get_team_comp(mgr_num, flex=flex)
        needed_pos_counts = {pos: STARTER_COMPOSITION[pos] - team_comp[pos] for pos in STARTER_COMPOSITION}
        return needed_pos_counts
    
        
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
        '''first two rounds snake, third round jumps back to last manager and snakes from there'''
        for i in range(12):
            turns.append({"round": 0, "mgr": i})
        for i in range(11, -1, -1):
            turns.append({"round": 1, "mgr": i})
            
        for round_num in range(2, 15):
            if round_num % 2 == 0:
                for i in range(11, -1, -1):
                    turns.append({"round": round_num, "mgr": i})
            else:  
                for i in range(12):
                    turns.append({"round": round_num, "mgr": i})
        return turns
    
    
    def reset_open_players(self):
        ''' Reset the open players to all players '''
        self.open_players = self.all_players.copy()
        self.open_players = self.open_players.loc[~self.open_players["sleeper_id"].isin([v["sleeper_id"] for v in self.keepers.values()])]
        
    def reset(self, seed=None):
        """Resets the environment to an initial state and returns an initial observation."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.draft = self._make_empty_draft()
        self.cur_round = 0
        self.cur_turn = 0
        self.dummy_round = False
        self.reset_open_players()
        
        
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
    
    def get_team(self, mgr_num: int, bye=False) -> pd.DataFrame:
        team = self.draft.loc[self.draft["mgr"] == mgr_num].copy()
        if bye:
            team = team.merge(self.bye_weeks, on="team", how="left")
            team.loc[team["position"] == "DEF", 'bye'] = np.nan
        return team
    
    def get_mgr_rankings(self, starters=True) -> pd.DataFrame:
        # identify which rounds are all full
        # full_rounds = self.draft.groupby('round')['mean'].apply(lambda x: x.notnull().all())
        # full_rounds = full_rounds[full_rounds].index
        # draft = self.draft[self.draft['round'].isin(full_rounds)]
        if starters:
            starter_draft = pd.concat([self.get_starters(mgr_num) for mgr_num in range(NUM_MGRS)])
            rankings = starter_draft.groupby('mgr')['mean'].sum().sort_values(ascending=False)
        else:
            rankings = self.draft.groupby('mgr')['mean'].sum().sort_values(ascending=False)
        rankings = pd.DataFrame({
            "rank": np.arange(1, NUM_MGRS+1),
            "mgr": rankings.index,
            "mean": rankings.values
            }).reset_index(drop=True)
        return rankings
    
    def calc_reward(self, mgr_num: int) -> float:
        
        '''
        Calculate the reward for a manager's choice 
        Must input a round for which you want the reward 
        because the reward takes into account who the manager drafted that round 
        i.e. if they drafted a starter
        
        '''
        
        starters = self.get_starters(mgr_num)
        bench = self.get_bench(mgr_num)
        
        starters_sum = starters[self.objective].sum()/10
        bench_sum = bench[self.objective].sum()/5


        reward = 0
        if self.is_starters_filled(mgr_num):
            reward = 1.25*starters_sum + .2*bench_sum
            # reward = starters_sum
        else:
            reward = -.5
            
        # -- bonus for first place -- #
        if self.get_mgr_rankings().iloc[0]["mgr"] == mgr_num:
            reward += 1
        # -- bonus for second place -- #
        if self.get_mgr_rankings().iloc[1]["mgr"] == mgr_num:
            reward += .5
        # -- bonus for third place -- #
        if self.get_mgr_rankings().iloc[2]["mgr"] == mgr_num:
            reward += .25
            
        # -- penalize double defense or double kicker -- #
        team_comp = self.get_team_comp(mgr_num, flex=False)
        if team_comp.get("DEF", 0) > 1:
            reward -= .5
        if team_comp.get("K", 0) > 1:
            reward -= .5
            
        # num_incomplete_team_weeks = self.get_num_incomplete_team_weeks(mgr_num)
        # reward -= num_incomplete_team_weeks/(17*2)
            
        # # -- clip rewards -- #
        # reward = np.clip(reward, a_min=-1, a_max=None).astype(np.float32).item()
            
        return reward
    
    
    def get_num_incomplete_team_weeks(self, mgr_num: int) -> int:
        ''' Get the number of weeks without full starter lineup '''
        team = self.get_team(mgr_num, bye=True)
        missing_player_weeks = 0
        for week in range(1, 18):
            week_team = team.copy()
            week_team = week_team.loc[week_team["bye"] != week]
            pos_counts = week_team.groupby("position").size()
            # first check flex
            flex_possible = pos_counts.get("RB", 0) + pos_counts.get("WR", 0) + pos_counts.get("TE", 0)
            flex_needed = STARTER_COMPOSITION["RB"] + STARTER_COMPOSITION["WR"] + STARTER_COMPOSITION["TE"]
            if flex_possible < flex_needed:
                missing_player_weeks += 1
                continue
            # Then check other positions
            for pos, count in STARTER_COMPOSITION.items():
                if pos == "FLEX":
                    continue
                if pos_counts.get(pos, 0) < count:
                    missing_player_weeks += 1
                    break
        return missing_player_weeks
                
            
        
    
    def choose_player(self, mgr_num: int, sleeper_id: str):
        '''
        updates draft by adding chosen player to and open players
        '''
        # identify keeper round for manager
        keeper_round = self.keepers[mgr_num]['round']
        
        # If there is no player and it is not the keeper round, return None
        # if sleeper_id is None or sleeper_id not in self.open_players["sleeper_id"].values:
        #     if int(keeper_round) != int(self.cur_round):
        #         return None

        if sleeper_id and sleeper_id in self.open_players["sleeper_id"].values and int(keeper_round) != int(self.cur_round):   
            # Get the player from the open players
            player = self.open_players.loc[self.open_players["sleeper_id"] == sleeper_id].iloc[0].to_dict()
        
        
        # Ignore choice and replace player with keeper if it is the keeper round for the manager
        if int(keeper_round) == int(self.cur_round):
            player_row = self.all_players.loc[self.all_players['sleeper_id'] == self.keepers[mgr_num]['sleeper_id']]
            player = {}
            for k, v in player_row.iloc[0].items():
                if k in self.draft.columns:
                    player[k] = v
            # print("---")
            # print(self.keepers[mgr_num]['sleeper_id'])
            # print(player['sleeper_id'])
            # print(self.open_players.shape)
            self.open_players = self.open_players.loc[self.open_players["sleeper_id"] != str(player['sleeper_id'])]
            # print(self.open_players.shape)
            # print("---")
        
        # If it is not keeper round and no player was given return no player
        if not sleeper_id or sleeper_id not in self.open_players["sleeper_id"].values and int(keeper_round) != int(self.cur_round):
            return None
        
        team_comp = self.get_team_comp(mgr_num, flex=False)
        # Indicates if the team is ready to draft a flex player, ie. rb, wr, te are filled
        flex_ready = team_comp.get("RB") >= STARTER_COMPOSITION["RB"] and \
            team_comp.get("WR") >= STARTER_COMPOSITION["WR"] and \
            team_comp.get("TE") >= STARTER_COMPOSITION["TE"]
        # If mgr is ready for flex player, then assign FLEX
        if player["position"] in ["RB", "WR", "TE"] and flex_ready:
            player["team_pos"] = "FLEX"
        else: 
            player["team_pos"] = player["position"]
        
        # -- Add the player to the draft -- #
        for k, v in player.items():
            if k in self.draft.columns:
                self.draft.loc[self.cur_turn, k] = v
        
        # -- Remove the player from the open players -- #
        self.open_players = self.open_players.loc[self.open_players["sleeper_id"] != str(player['sleeper_id'])]
        
        return player
    
        
    
    def increment_turn(self):
        '''updates the state, increments the turn, and updates the round'''
        self.update_state()
        
        self.cur_turn += 1
        if self.cur_turn < len(self.turns):
            # Normal round updating
            self.cur_round = self.draft["round"].values[self.cur_turn]
            self.cur_mgr = self.draft["mgr"].values[self.cur_turn]
            return True
        
        # elif self.cur_turn == len(self.turns):
        #     # start of dummy round
        #     self.dummy_round = True
        #     self.cur_round = NUM_DRAFT_ROUNDS
        #     self.cur_mgr = 0
        #     return True
        # else:
            # continuing dummy round 
            # self.cur_round = NUM_DRAFT_ROUNDS
            # self.cur_mgr += 1
        else:
            self.cur_round = NUM_DRAFT_ROUNDS
            self.cur_mgr = None
            

    def stochastic_choice(self, mgr_num: int=None, temperature=None) -> dict:
        '''
        picks a random (intelligent) player for the NPC managers.
        unlike reasonable_action methods, this returns the player as a dict
        This allows us to choose players who are not the highest ranked player in a position
        
        First, draft QB, RB, WR, TE to fill starters incl. flex.
        Do this by taking the top 5 players in needed positions and
        sampling from their softmaxed means.
        That takes 8 rounds.
        In rounds 9-13, K and DEF will be included in softmax.
        In rounds 14-15, verify that every position is filled.
        
        '''
        if temperature is None:
            temperature = self.stochastic_temp
        if mgr_num is None:
            mgr_num = self.get_cur_mgr()
        team_comp = self.get_team_comp(mgr_num, flex=True)
        
        if self.cur_round < 8: # rounds 1-8
            needed_pos_counts = self.get_needed_pos_counts(mgr_num, flex=True)
            needed_pos_counts["K"] = 0
            needed_pos_counts["DEF"] = 0
            # first get what non-flex non kicker non def positions are needed
            needed_positions = [pos for pos, count in needed_pos_counts.items() if pos != "FLEX" and count > 0]
            # check if rb, wr, te are filled so we can move to working on flex
            rb_te_wr_filled = team_comp.get("RB", 0) >= STARTER_COMPOSITION["RB"] and \
                team_comp.get("WR", 0) >= STARTER_COMPOSITION["WR"] and \
                team_comp.get("TE", 0) >= STARTER_COMPOSITION["TE"]
                
            # if the rb, wr, and te are filled, check if flex is filled
            # if flex is not filled, add rb, wr, te to needed positions
            if needed_pos_counts.get("FLEX", 0) > 0 and rb_te_wr_filled:
                    needed_positions += ["RB", "WR", "TE"]
            
            # Now we have established which positions to look for 
            # if we don't need players or if no players are open in needed position, 
            # it will sample from the top 5 players
            player = self._sample_player(temperature=temperature, needed_positions=needed_positions)
            return player

        if self.cur_round >= 8 and self.cur_round < 13: # rounds 9-13
            # in rounds 8-13, go for non-K and non-DEF positions alongside K and DEF
            # and then once those are filled, go for K and DEF alongside all players
            needed_pos_counts = self.get_needed_pos_counts(mgr_num, flex=True)
            needed_positions = [pos for pos, count in needed_pos_counts.items() if pos != "FLEX" and count > 0]
            rb_te_wr_filled = team_comp.get("RB", 0) >= STARTER_COMPOSITION["RB"] and \
                team_comp.get("WR", 0) >= STARTER_COMPOSITION["WR"] and \
                team_comp.get("TE", 0) >= STARTER_COMPOSITION["TE"]
            if needed_pos_counts.get("FLEX", 0) > 0 and rb_te_wr_filled:
                needed_positions += ["RB", "WR", "TE"]
            exclude = [pos for pos in ["K", "DEF"] if team_comp.get(pos, 0) >= STARTER_COMPOSITION[pos]]
            if any([pos != "K" and pos != "DEF" for pos in needed_positions]):
                # if anything is needed besides K and DEF
                player = self._sample_player(temperature=temperature, needed_positions=needed_positions, exclude_pos=exclude)
            else:
                player = self._sample_player(temperature=temperature, exclude_pos=exclude)
            return player
        
        if self.cur_round >= 13: # rounds 14-15
            needed_pos_counts = self.get_needed_pos_counts(mgr_num, flex=True)
            needed_positions = [pos for pos, count in needed_pos_counts.items() if pos != "FLEX" and count > 0]
            rb_te_wr_filled = team_comp.get("RB", 0) >= STARTER_COMPOSITION["RB"] and \
                team_comp.get("WR", 0) >= STARTER_COMPOSITION["WR"] and \
                team_comp.get("TE", 0) >= STARTER_COMPOSITION["TE"]
            if needed_pos_counts.get("FLEX", 0) > 0 and rb_te_wr_filled:
                 needed_positions += ["RB", "WR", "TE"]
            exclude = [pos for pos in ["K", "DEF"] if team_comp.get(pos, 0) >= STARTER_COMPOSITION[pos]]
            # print(self.cur_round)
            # print(exclude)
            # print(team_comp)
            # print(needed_positions)
            player = self._sample_player(temperature=temperature, needed_positions=needed_positions, exclude_pos=exclude)
            return player
        
    
    
    def _sample_player(self, temperature, needed_positions=None, exclude_pos=None, samp_size=5):
        ''' 
        Choose a player from a list of needed positions 
        If none are available, sample from the top 5 players (within 100 points of the top player)
        Downweight probability of defense since points are higher but variance is super high and preds are high
        '''
        if needed_positions:
            # print(needed_positions)
            players = self.open_players[self.open_players['position'].isin(needed_positions)].copy()
        if not needed_positions or players.empty:
            players = self.open_players.copy()
        if exclude_pos:
            players = players.loc[~players["position"].isin(exclude_pos)]
        options = players.loc[players["mean"] >= players["mean"].max() - 100]
        options = options.iloc[:np.min([samp_size, len(options)])]
        options.loc[options["position"] == "DEF", "mean"] -= 25  # Downweighting defense 
        w = softmax(options["mean"].values, temperature=temperature)
        chosen_player = options.sample(1, weights=w).iloc[0]
        return chosen_player.to_dict()
  
    
    
    def step(self, action, deterministic=True):
        
        # print("8146" in self.open_players["sleeper_id"].values)
        cur_round = self.cur_round
        cur_turn = self.cur_turn
        assert cur_turn < len(self.draft), "Draft is over"
        # mgr_num = self.cur_mgr
        mgr_num = self.draft["mgr"].values[cur_turn]
        # assert cur_turn <= len(self.draft), "Draft is over"  # added dummy round
        
        # -- Check if this is last pick of draft -- #
        # terminated = True if cur_turn == len(self.draft)-1 else False
        
        # -- Choose the player -- #
        # print("8146" in self.open_players["sleeper_id"].values)
        filtered_players = self.open_players.loc[self.open_players["position"] == self.actions[action]]
        filtered_players = filtered_players.sort_values(by="mean", ascending=False).reset_index(drop=True) 
        if not filtered_players.empty:
            if deterministic:
                chosen_player = filtered_players.iloc[0]
            else:
                # my concern with this method is that it will make choices where the next three players have less differential
                # a better method would be passing in the action probabilities and sampling from that
                # w = softmax(filtered_players["mean"].values, temperature=.1)
                # chosen_player = filtered_players.sample(weights=w).iloc[0]
                # would be good to log and check whether these weights are good
                raise NotImplementedError
                
        else:
            # If no players are available for the chosen position, they get no player
            chosen_player = {k: None for k in filtered_players.columns}
            chosen_player["mean"] = 0
            chosen_player["std"] = 0
            
        # -- update the draft and open players -- #
        # (player could be different if it is the keeper round)
        actual_player = self.choose_player(mgr_num, chosen_player["sleeper_id"])
        
        # -- determine if round is over and then calculate reward -- #
        # if (self.cur_turn+1) % NUM_MGRS == 0:
        
        # -- determine if draft is over and then calculate reward -- #
        # if terminated:
        #     reward = [self.calc_reward(mgr_num) for mgr_num in range(NUM_MGRS)]
        # else:
        #     reward = [0 for mgr_num in range(NUM_MGRS)]
            
        
        # -- increment the turn and round and update state vector -- #
        turn = self.cur_turn
        self.increment_turn()
        
        
        # -- calculate reward -- #
        # NOTE THIS DOES NOT EXACTLY WORK IF NOT DOING SARL BECAUSE DRAFT IS NOT OVER WHEN THIS GETS CALLED FOR MARL
        # THAT SAID, IT KIND OF WORKS FOR MARL UNLESS SOMEONE DRAFTS STARTER IN LAST ROUND
        # reward = self.calc_reward(mgr_num, cur_round) # calculates reward without first place bonus
        
        # -- store the draft for step output -- #
        draft = self.draft.to_dict(orient="records")
        
        info = {
            "cur_turn": turn,
            "cur_round": cur_round,
            "player": actual_player,
            "mgr_num": mgr_num,
            # "terminated": terminated,
            # "reward": reward,
            "notes": None,
            "draft": draft
        }  
        
        
        truncated = False
        terminated = False
        reward = 0  # reward is always 0 from step bc it is calculated in the next step
        
        return self.state, reward, terminated, truncated, info
    
    def draft_full(self) -> bool:
        return self.draft["mean"].isnull().sum() == 0
    
    def get_mgr_draft(self, mgr_num: int) -> pd.DataFrame:
        return self.draft.loc[self.draft["mgr"] == mgr_num]
    
    def get_state(self):
        return self.state
    
    def get_sum_fp(self, mgr_num: int, starters=False) -> float:
        # draft = self.draft.loc[self.draft["round"] <= self.cur_round]
        
        # # Identify combinations of 'mgr' and 'round' where 'mean' is fully filled
        # grouped = draft.groupby(['mgr', 'round'])['mean'].apply(lambda x: x.notnull().all())

        # # Filter the DataFrame to include only the valid combinations
        # completed_rounds = grouped[grouped].index
        # draft = draft[draft.set_index(['mgr', 'round']).index.isin(completed_rounds)].reset_index(drop=True)
        
        # # Group by 'mgr' and sum 'mean', converting the result to a dictionary
        # sum_fp = draft.groupby("mgr")["mean"].sum().to_dict(orient='records')
        # sum_fp['round'] = draft['round'].max()
        
        # If 'starters' is True, filter the DataFrame to include only the starters
        if starters:
            team = self.get_starters(mgr_num)
        else: 
            team = self.get_team(mgr_num)
            
        sum_fp = team["mean"].sum()
    
        return sum_fp
    
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
        print(self.draft)
    
    def close(self):
        """Cleanup the environment before closing (optional)."""
        pass
    

