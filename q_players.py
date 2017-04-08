"""
Created on Sat Apr  1 20:23:14 2017

@author: matthew
"""

__author__ = "Matthew Deakos"

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
            episode_list: Keeps track of state action pairs used in the current training episode
        """
        super().__init__()
        self.Q_Table = defaultdict(dict)
        self.N_Table = defaultdict(dict)
        self.N_nought = 100
        self.learning = True
        self.episode_list = []

        
    def get_Q_value(self,state):
        """Retrives the Q value from a given state or state_key
        """
        if isinstance(state,dict):
            state_key = tuple(state.values())
        elif isinstance(state,tuple):
            state_key = state
        else:
            raise ValueError("{} not a valid state".format(state))
        
        action_dict = self.Q_Table.get(state_key)
        
        if action_dict is None or len(action_dict) == 0:
            return 0
        return max(action_dict.values())
        

    def choose_action(self,state,rand=False):
        """Chooses an action.
        A heuristic is implemented in the case of the Easy21 environment
        due to the size of the potential state space. The rule is that
        the agent should hit if the total is under 21
        """
        entry = self.Q_Table.get(state)
    
        if rand:
            action = random.choice(self.environment.valid_actions)
        else:
            best_actions = [key for key,item in entry.items() if item == max(entry.values())]
            action = random.choice(best_actions)
        
        self.N_Table[state][action] += 1
        return action
        
        
    def load_Q_Table(self,q_table):
        """Loads a Q Table as a dictionary of states to action dictionaries
        """
        self.Q_Table = q_table
        self.learning = False
        
        
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
        """Generates a Q_Table entry for a state and initializes all actions to a Q_Value of 0"""
        
        action_dict = {}
        for action in self.environment.valid_actions:
            action_dict[action] = 0
        self.Q_Table[state] = action_dict
        self.N_Table[state] = action_dict.copy()
    
   
    def act(self,state):
        """Act in the environment.
        
        Parameters:
            state: A state dictionary from the environment
            
        Returns:
            reward: The reward gained from acting in the environment.
        """
       
        state_key = tuple(state.values())
        if self.Q_Table.get(state_key) is None:
            self.generate_Q_entry(state_key)
            
        if self.learning:
            rolled_epsilon = random.random()
            state_epsilon = self.get_epsilon(state_key)
        
        if self.learning and rolled_epsilon < state_epsilon:
            action = self.choose_action(state_key,rand=True)
        else:
            action = self.choose_action(state_key)
                        
        next_state,reward = self.environment.step(action)
        
        self.episode_list.append((state_key,action))
        
        if next_state == None and self._learning:
            self.update_Q_Table(reward)
            self.episode_list = []
            
        return reward
            
    def update_Q_Table(self,reward):
        """Updates Q(S,a)"""
        for state,action in self.episode_list:
            visit_count = self.N_Table[state][action]
            current_Q = self.Q_Table[state][action]
            error = (reward - current_Q)
            alpha = 1/float(visit_count) #Decaying alpha
                    #print(alpha)
                    #print(reward)
            self.Q_Table[state][action] = current_Q + alpha*error
            
class QLearner(Basic_Player):
    """Q-Learner
    
    Monte-Carlo Q-Learning algorithm with a linear function approximator.
    """
    
    def __init__(self,alpha=0.01,epsilon=0.05):
        """Initializer for the Q-Learner
        
        Attributes:
            weights: weight matrices used in linear function approximation. One matrix per viable action.
            episodes: a list of state-action pairs since last update
            alpha: learning rate [0-1]
            epsilon: e-greedy exploration parameter [0-1]
        """
        super().__init__()
        self.weights = {}
        self.episodes = []
        self.alpha = alpha
        self.epsilon = epsilon

    def generate_feature_vector(self,state):
        """Turns the state dictionary into a feature vector.
        
        The feature vector is coarsely coded such that every 2 consecutive state values
        are represented by one state.
        
        Example:
        Player Sum (10) and Player Sum(11) both represent the same state.
        
        """ #TODO: Clarify this docstring
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
         
        #Generate weights if they have not yet been initialized.
        if (self.weights == {}):
            self._generate_weights(feature_vector)
        
        return feature_vector
    
    def get_Q_value(self,feature_vector,action):
        """Returns the Q-Value of a state-action pair."""
        return np.dot(self.weights[action],feature_vector)
    
    def get_gradient(self,feature_vector,action,true):
        """Returns the gradient of a state-action pair"""
        q_val = self.get_Q_value(feature_vector,action)
        error = q_val - true
        return np.multiply(error,feature_vector)
        
    def update_value(self,feature_vector,action,reward):
        """Updates the Q-value of a state-action pair according to the learning rate"""
        gradient = self.get_gradient(feature_vector,action,reward)
        update = np.multiply(self.alpha,gradient)
        self.weights[action] -= update
        
    def act(self,state):
        """Determines the action to be taken given the current state"""
        feature_vector = self.generate_feature_vector(state)
        action = self.choose_action(feature_vector,self.epsilon)
        
        self.episodes.append((feature_vector,action))
        
        next_state,reward = self.environment.step(action)
        
        if next_state is None:
            self.update_Q_function(reward)
            
        return reward
        
    def update_Q_function(self,reward):
        """Monte Carlo Update"""
        for feature_vector,action in self.episodes:
            self.update_value(feature_vector,action,reward)
        self.episodes = []
            
    
    def choose_action(self,feature_vector,epsilon):
        """Chooses an action according to an epsilon-greedy Q-value lookup"""
        if epsilon <= np.random.random():
            q_vals = {}
            for action in self.environment.valid_actions:
                q_value = self.get_Q_value(feature_vector,action)
                q_vals[action] = q_value  
            best_actions = [key for key,q in q_vals.items() if q == max(q_vals.values())]
            chosen_action = random.choice(best_actions)   
        else:
            chosen_action = np.random.choice(self.environment.valid_actions)
        return chosen_action
        
    def _generate_weights(self,feature_vector):
        """Generates weights based on the size of the feature vector"""
        for action in self.environment.valid_actions:
            weights = np.zeros([len(feature_vector)])
            self.weights[action] = weights
            
            
class SARSALearner(Basic_Player):
    """SARSA(lambda) learner with a Linear Function Approximation"""
    
    def __init__(self,alpha=0.01,epsilon=0.05,gamma=0.9,lmbda = 1):
        """Initializes the SARSA agent.
        
        Attributes:
            weights: weights used by the linear function approximator. One weight matrix per action.
            epsilon: epsilon value for an epsilon greedy improvement policy (0-1)
            alpha: learning rate [0-1]
            lmbda: lambda for SARSA(lambda) improvement policy - determines strength of eligibility trace [0-1]
            eligibility: eligibility table. One per action.
            gamma: Decay rate of reward
            action: Next action to be taken in the environment
        """
        super().__init__()
        self.weights = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.lmbda = lmbda
        self.eligibility = {}
        self.gamma = gamma
        self.action = None
        
    def act(self,state):
        """Chooses an action according to the SARSA lambda algorithm
        """
        ##Turn the state into a feature vector
        feature_vector = self.generate_feature_vector(state)
        
        #Choose an action if this is the first state in the episode
        if self.action is None:
            self.action = self.choose_action(feature_vector)
        
        next_state,reward = self.environment.step(self.action)
        
        self.SARSA_update(feature_vector,next_state,reward)

        return reward
        
    def generate_feature_vector(self,state):
        """Turns the state dictionary into a feature vector.
        
        The feature vector is coarsely coded such that every 2 consecutive state values
        are represented by one state.
        
        Example:
        Player Sum (10) and Player Sum(11) both represent the same state.
        
        """ #TODO: Clarify this docstring
        
        feature_vector = np.zeros([30])
        player_val = state['p_sum']
        dealer_val = state['d_start']
        player_state = 0
        
        if player_val % 2 == 0:
            player_state = int(((player_val - 1) % 11) / 2)
        else:
            player_state = int((player_val % 11) / 2)
            
        if dealer_val % 2 == 0:
            dealer_state = int((dealer_val - 1) / 2)
        else:
            dealer_state = int(dealer_val / 2)
        
        feature_vector[dealer_state * 6 + player_state] = 1 
            
        ##Initialize Weights 
        if (self.weights == {}):
            self._generate_weights(feature_vector)
        
        return feature_vector
    
    def get_Q_value(self,feature_vector,action):
        """Returns the Q_value of a state-action pair"""
        return np.dot(self.weights[action],feature_vector)
       
    def SARSA_update(self,feature_vector,next_state,reward):
        """Updates the Q_function according to the SARSA update algorithm"""
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
       
       
    def choose_action(self,feature_vector):
        """Chooses an action based on an epsilon-greedy SARSA policy"""
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
     
     
    def _generate_weights(self,feature_vector):
        """Generates the weights for the linear function approximator"""
        for action in self.environment.valid_actions:
            weights = np.zeros([len(feature_vector)])
            eligibility = np.zeros([len(feature_vector)])
            self.weights[action] = weights
            self.eligibility[action] = eligibility