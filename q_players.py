# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 20:23:14 2017

@author: matthew
"""
from player import Basic_Player
from collections import defaultdict
from numpy import random
import numpy as np

try:
    from environment import Easy21_Environment
except ImportError:
    pass #Do nothing if the easy21 environment cannot be imported


class QLearner_Basic(Basic_Player):
    """This QLearner uses the Monte Carlo method and a Q-Table.
    """
    def __init__(self):
        """Initialization
        
        Attributes:
            Q_Table: Dictionary of State Action Pairs and their Q-Value
            N_Table: Dictionary of State Action Pairs and their visit count
            N_Nought: Value used in epsilon calculation (see get_epsilon)
            log_file: Keeps track of actions and rewards
            episode_list: Keeps track of state action pairs used in the current training episode
        """
        super().__init__()
        self.Q_Table = defaultdict(dict)
        self.N_Table = defaultdict(dict)
        self.N_nought = 100
        self._learning = True
        self.log_file = defaultdict(int)
        self.episode_list = []
        
    def setLearning(self,learning):
        """Set the learning to false
        """
        self._learning = learning
        
    #Choose action override
    def choose_action(self,state,rand=False):
        """Chooses an action.
        A heuristic is implemented in the case of the Easy21 environment
        due to the size of the potential state space. The rule is that
        the agent should hit if the total is under 21
        """
        entry = self.Q_Table.get(state)
        
        if state[1] < 11 and isinstance(self.environment,Easy21_Environment):
            action = "hit"
        else:
            if rand:
                action = random.choice(self.valid_actions)
            else:
                best_actions = [key for key,item in entry.items() if item == max(entry.values())]
                action = random.choice(best_actions)
        
        self.N_Table[state][action] += 1
        return action
        
    def load_Q_Table(self,q_table):
        """Loads a Q Table as a dictionary of states to action dictionaries
        """
        self.Q_Table = q_table
        self._learning = False
        
    def get_epsilon(self,state):
        """Calculates epsilon as N_nought / (N_nought + state_visits)
        """
        n_entry = self.N_Table.get(state)
        if n_entry is None:
            state_visits = 0
        else:
            state_visits = np.sum(list(n_entry.values()))
            
        return self.N_nought / float((self.N_nought + state_visits))
    
    def generate_Q_entry(self,state):
        """Generates a Q_Table entry for a state and initializes
        all actions to a Q_Value of 0
        """
        action_dict = {}
        for action in self.valid_actions:
            action_dict[action] = 0
        self.Q_Table[state] = action_dict
        self.N_Table[state] = action_dict.copy()
    
    #Q Learning via Monte Carlo method
    def act(self,state):
        """
        """
        state_key = tuple(state.values())
        if self.Q_Table.get(state_key) == None:
            self.generate_Q_entry(state_key)
            
        
        if self._learning:
            rolled_epsilon = random.random()
            state_epsilon = self.get_epsilon(state_key)
        
        action_choice = ""
        
        if self._learning and rolled_epsilon < state_epsilon:
            action_choice = "Random"
            action = self.choose_action(state_key,rand=True)
        else:
            action_choice = "Greedy"
            action = self.choose_action(state_key)
            
        self.log_file[(state_key,action,action_choice)] += 1
            
        next_state,reward = self.environment.step(action)
        
        self.episode_list.append((state_key,action))
        
        if next_state == None and self._learning:
            self.update_Q_Table(reward)
            self.episode_list = []
            
            
        return next_state,reward
            
    def update_Q_Table(self,reward):
        for state,action in self.episode_list:
            visit_count = self.N_Table[state][action]
            current_Q = self.Q_Table[state][action]
            error = (reward - current_Q)
            alpha = 1/float(visit_count)
                    #print(alpha)
                    #print(reward)
            self.Q_Table[state][action] = current_Q + alpha*error
            
            
class QLearner(Basic_Player):
    
    def __init__(self):
        super().__init__()