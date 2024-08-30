
from collections import deque, namedtuple
from typing import Dict, List
import random
import numpy as np
import pandas as pd
import math
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_SPACE_DIM = 6
STATE_SCHEMA = {"QB_mean": 15, "RB_mean": 30, "WR_mean": 30, "TE_mean": 15, "K_mean": 15, "DEF_mean": 15, 
                "round": 1, "turns_until_next_pick": 1,
                "mgr_0_team": 6, "mgr_1_team": 6, "mgr_2_team": 6,
                "mgr_3_team": 6, "mgr_4_team": 6, "mgr_5_team": 6,
                "mgr_6_team": 6, "mgr_7_team": 6, "mgr_8_team": 6, 
                "mgr_9_team": 6, "mgr_10_team": 6, "mgr_11_team": 6
                }

STARTER_COMPOSITION = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1, "FLEX": 2}
STATE_SPACE_DIM = np.sum(list(STATE_SCHEMA.values()))
NUM_DRAFT_ROUNDS = 15
NUM_MGRS = 12



class DQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim, hidden_dim=64, hidden_layers=2):
        super(DQN, self).__init__()
        self.layer_in = nn.Linear(state_space_dim, hidden_dim)
        # make hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layer_out = nn.Linear(hidden_dim, action_space_dim)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.layer_out(x)
        return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # upweight samples with reward > 0
        # make mask for positive rewards
        # pos = [i for i in range(len(self.memory)) if self.memory[i].reward > 0]
        # neg = [i for i in range(len(self.memory)) if self.memory[i].reward <= 0]
        # pos_weight = 1.0
        # neg_weight = 1.0
        # if len(pos) > 0:
        #     pos_weight = len(self.memory) / len(pos)
        # if len(neg) > 0:
        #     neg_weight = len(self.memory) / len(neg)
        # mask = [pos_weight if i in pos else neg_weight for i in range(len(self.memory))]
        # mask = np.array(mask)
        # mask = mask / np.sum(mask)
        # return random.choices(self.memory, k=batch_size, weights=mask)
    
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
