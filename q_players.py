# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 20:23:14 2017

@author: matthew
"""
from player import Basic_Player
from collections import defaultdict
from numpy import random
import numpy as np
import traceback

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
    """SARSA via Linear Function Approximation
    """
    
    def __init__(self,alpha=0.01,epsilon=0.05,gamma=0.9,lmbda = 1):
        super().__init__()
        self.weights = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.lmbda = lmbda
        self.eligibility = {}
        self.gamma = gamma
        self.action = None

    def _generate_weights(self,feature_vector):
        for action in self.environment.valid_actions:
            weights = np.zeros([len(feature_vector)])
            eligibility = np.zeros([len(feature_vector)])
            self.weights[action] = weights
            self.eligibility[action] = eligibility


    def generate_feature_vector(self,state):
        feature_vector = np.zeros([30])
        player_val = state['p_sum']
        dealer_val =state['d_start']
        player_state = 0
        if player_val % 2 == 0:
            player_state = int(((player_val - 1) % 11)/2)
        else:
            player_state = int((player_val % 11)/2)
            
        if dealer_val % 2 == 0:
            dealer_state = int((dealer_val - 1)/2)
        else:
            dealer_state = int(dealer_val/2)
        
        feature_vector[dealer_state * 6 + player_state] = 1 
            
        ##Initialize Weights 
        if (self.weights == {}):
            self._generate_weights(feature_vector)
        
        return feature_vector
    
    def get_Q_value(self,feature_vector,action):
        return np.dot(self.weights[action],feature_vector)
        

    def SARSA_update(self,feature_vector,next_state,reward):
        ##Get the Q_value in the current state
        q_current = self.get_Q_value(feature_vector,self.action)
        
        ##If in the terminal state, the reward is all that matters
        if next_state is None:
            q_next = 0
            next_action = None
        else:
            next_feature_vector = self.generate_feature_vector(next_state)
            next_action = self.choose_action(next_feature_vector)
            q_next = self.get_Q_value(next_feature_vector,next_action)
        
        error = reward + self.gamma*q_next - q_current
        self.eligibility[self.action][np.where(feature_vector != 0)] += 1
        
        gradient = error*(-self.eligibility[self.action])
        update = np.multiply(self.alpha,gradient)
            
        #Update Weights
        self.weights[self.action] -= update
        
        #Update Eligibility Table
        self.eligibility[self.action] = self.gamma*self.lmbda*self.eligibility[self.action]
        self.action = next_action
        
        if next_state is None:
            #Reset eligibility table if in terminal state
            for action in self.environment.valid_actions:
                self.eligibility[action] = np.zeros(self.eligibility[action].size)
        
        
    def act(self,state):
        ##Turn the state into a feature vector
        feature_vector = self.generate_feature_vector(state)
        
        #Choose an action if this is the first state in the episode
        if self.action is None:
            self.action = self.choose_action(feature_vector)
        
        
        next_state,reward = self.environment.step(self.action)
        
        self.SARSA_update(feature_vector,next_state,reward)

            
        return next_state,reward
        
    
    
    def choose_action(self,feature_vector):
  
        if self.epsilon <= np.random.random():
            q_vals = {}
            for action in self.environment.valid_actions:
                q_value = self.get_Q_value(feature_vector,action)
                q_vals[action] = q_value
                
            best_actions = [key for key,q in q_vals.items() if q == max(q_vals.values())]
            chosen_action = random.choice(best_actions)
        else:
            chosen_action = np.random.choice(self.environment.valid_actions)
        return chosen_action
    
        
if __name__ == "__main__":
    environment = Easy21_Environment()
    agent = QLearner()
    environment.add_primary_agent(agent)
    results = []
    for i in range(1000):
        _,_,result = environment.play_game()
        results.append(result)
        
    test_state = {'p_sum':21,'d_start':8}
    f = agent.generate_feature_vector(test_state)
    q_val_hit = agent.get_Q_value(f,'hit')
    q_val_stick = agent.get_Q_value(f,'stick')
    print("Result: {}".format(sum(results)))
    
    print(q_val_hit)
    print(q_val_stick)