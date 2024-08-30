
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

from rl_env import (
    DraftEnv, softmax,
    ACTION_SPACE_DIM, STATE_SCHEMA,
    STARTER_COMPOSITION, STATE_SPACE_DIM,
    NUM_DRAFT_ROUNDS, NUM_MGRS
    )


class SARLDraftEnv(DraftEnv):
    """
    Custom Environment that follows gymnasium interface
    This version is explicitly designed for single-agent reinforcement learning
    I.e. each step of the environment does the picks for all managers
    
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, stochastic_temp, rl_mgr=1):
        super(SARLDraftEnv, self).__init__(stochastic_temp=stochastic_temp)
        self.rl_mgr = rl_mgr  # which manager is the RL agent
        
    def reset(self, seed=None):
        '''needs to reset the draft and then run it until the RL agent's turn'''
        state, info = super(SARLDraftEnv, self).reset(seed=seed)
        state = self.run_other_mgrs()
        return state, info
        
        
    def run_other_mgrs(self):
        '''runs the draft for all managers except the RL agent'''
        while self.cur_turn < len(self.draft) and self.get_cur_mgr() != self.rl_mgr:
            choice = self.stochastic_choice(temperature=self.stochastic_temp)
            self.choose_player(self.get_cur_mgr(), choice['sleeper_id'])
            self.increment_turn()
        return self.state
    
    def between_turns_actions(self):
        return self.run_other_mgrs()

    
    # def step(self, action):
        
    #     assert self.get_cur_mgr() == self.rl_mgr, "It is not the RL agent's turn"
        
    #     cur_round = self.cur_round
        
    #     mgr_num = self.draft["mgr"].values[self.cur_turn]
        
    #     terminated = True if self.cur_round == NUM_DRAFT_ROUNDS - 1 else False
        
    #     # self.update_state()  # added this in when i was going to have the roster in the state
        
    #     # -- Choose the player -- #
    #     filtered_players = self.open_players.loc[self.open_players["position"] == self.actions[action.item()]]
    #     filtered_players = filtered_players.sort_values(by="mean", ascending=False).reset_index(drop=True) 
    #     if not filtered_players.empty:
    #         chosen_player = filtered_players.iloc[0]
    #     else:
    #         # If no players are available for the chosen position, they get no player
    #         chosen_player = {"mean": 0, "std": 0, 
    #                          "sleeper_id": None, "full_name": None, 
    #                          "team": None, "position": None}  # or handle the case where no player is found

    #     # update the draft and open players
    #     actual_player = self.choose_player(mgr_num, chosen_player["sleeper_id"])
    #     reward = self.calc_reward(mgr_num, cur_round)
    #     draft = self.draft.to_dict(orient="records")
    #     turn = self.cur_turn
    #     self.increment_turn()
        
    #     truncated = False
        
    #     self.run_other_mgrs()
        
        
    #     info = {
    #         "cur_turn": turn,
    #         "cur_round": cur_round,
    #         "player": actual_player,
    #         "mgr_num": mgr_num,
    #         "terminated": terminated,
    #         "reward": reward,
    #         "notes": None,
    #         "draft": draft
    #     }  
        
    #     if terminated:
    #         rankings = self.get_mgr_rankings()
    #         first_place = rankings.index[0] == mgr_num
    #         reward += 1 if first_place else 0
        
    #     return self.state, reward, terminated, truncated, info
    
    


