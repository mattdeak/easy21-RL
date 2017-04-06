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
        """Decision on acting
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
    """Monte Carlo Learner via Linear Function Approximation
    """
    
    def __init__(self,alpha=0.001):
        super().__init__()
        self.weights = None
        self.episodes = []
        self.N_Table = defaultdict(dict)
        self.epsilon = 1.0
        self.alpha = alpha
        
    def _generate_weights(self,feature_vector):
        self.weights = np.zeros([feature_vector.size])


    def generate_feature_vector(self,state_vector,action):
        values = list(state_vector)
        values.append(self.environment.valid_actions.index(action))
        feature_vector = np.array(values)
        
        if (self.weights is None):
            self._generate_weights(feature_vector)
        
        return feature_vector
    
    def get_Q_value(self,feature_vector):
        return np.dot(self.weights,feature_vector)
    
    def get_gradient(self,feature_vector,true):
        q_val = self.get_Q_value(feature_vector)
        error = q_val-true
        print("Feature Vector:{}\nQ Val: {}\nError: {}".format(feature_vector,q_val,error))
        return np.multiply(error,feature_vector)
        
    def update_value(self,feature_vector,reward):
        gradient = self.get_gradient(feature_vector,reward)
        print("Feature Vector: {}\nGradient: {}".format(feature_vector,gradient))
        self.weights -= np.multiply(self.alpha,gradient)
        
    def act(self,state):
        state_vector = list(state.values())
        epsilon = self.get_epsilon()
        action = self.choose_action(state_vector,epsilon)
        print(action,self.weights)
        feature_vector = self.generate_feature_vector(state_vector,action)
        if state_vector[0] >= 11:        
            self.episodes.append(feature_vector)
        
        next_state,reward = self.environment.step(action)
        
        if next_state is None:
            self.update_Q_function(reward)
            
        return next_state,reward
        
    def update_Q_function(self,reward):
        """Monte Carlo Update"""
        print(reward)
        for feature_vector in self.episodes:
            self.update_value(feature_vector,reward)
        self.episodes = []
            
    
    def choose_action(self,state_vector,epsilon):
        if state_vector[0] < 11:
            chosen_action = "hit"
        else:
            if epsilon >= np.random.random():
                q_vals = {}
                for action in self.environment.valid_actions:
                    feature_vector = self.generate_feature_vector(state_vector,action)
                    q_value = self.get_Q_value(feature_vector)
                    q_vals[action] = q_value
                
                best_actions = [key for key,q in q_vals.items() if q == max(q_vals.values())]
                chosen_action = random.choice(best_actions)
            else:
                chosen_action = np.random.choice(self.environment.valid_actions)
        return chosen_action
    
    def get_epsilon(self):
        epsilon = self.epsilon
        self.epsilon = min(0.0005,self.epsilon-0.0005)
        return epsilon
        
if __name__ == "__main__":
    environment = Easy21_Environment()
    agent = QLearner()
    environment.add_primary_agent(agent)
    for i in range(1000):
        _,_,result = environment.play_game()