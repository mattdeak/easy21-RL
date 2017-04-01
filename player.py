from environment import Environment
from numpy import random
import numpy as np
from collections import defaultdict

class Basic_Player:
    
    def __init__(self,debug=False):
        self.environment = None
        self.valid_actions = None
        self._debug = False
        
    def add_environment(self,environment):
        self.environment = environment
        self.valid_actions = environment.valid_actions
        
    def choose_action(self,state):
        action = random.choice(self.valid_actions)
        
        if self._debug:
            print(state['p_sum'],action)
        return action
        
    def act(self,state):
        if self.environment == None:
            raise ValueError("Must add an environment in order to act")
        
        #Choose an action randomly
        action = self.choose_action(state)
        
        next_state,reward = self.environment.step(action)        
        return reward
        
class Naive_Player(Basic_Player):
    
    #Override act
    def choose_action(self,state):
        if state['p_sum'] >= 18:
            action = 'stick'
        else:
            action = 'hit'
            
        if self._debug:
            print(state['p_sum'],action)
            
        return action
            
         
class Manual_Player(Basic_Player):
    
    #Override
    def choose_action(self,state):
        print("Current State: n {}".format(state))
        action = None
        while action == None:
            print("Valid Actions are: ")
            print(",".join(self.valid_actions))
            choice = input("Choose your action: ")
            if choice in self.valid_actions:
                action = choice
            else:
                print("Invalid choice")
        return action
        
    #Override
    def act(self,state):
        r = super().act(state)
        print("Reward: {}".format(r))
        return r
        
        
class QLearner(Basic_Player):
    
    def __init__(self):
        super().__init__()
        self.Q_Table = defaultdict(dict)
        self.N_Table = defaultdict(dict)
        self.N_nought = 100
        self._learning = True
        
    def setLearning(learning):
        self._learning = learning
        
    
    #Choose action override
    def choose_action(self,state,rand=False):
        entry = self.Q_Table.get(state)
        if rand:
            action = random.choice(self.valid_actions)
        else:
            best_actions = [key for key,item in entry.items() if item == max(entry.values())]
            action = random.choice(best_actions)
        self.N_Table[state][action] += 1
        return action
        
    def load_Q_Table(self,q_table):
        self.Q_Table = q_table
        self._learning = False
        
    def get_epsilon(self,state):
        n_entry = self.N_Table.get(state)
        if n_entry is None:
            state_visits = 0
        else:
            state_visits = np.sum(list(n_entry.values()))
            
        return self.N_nought / float((self.N_nought + state_visits))
    
    def generate_Q_entry(self,state):
        
        action_dict = {}
        for action in self.valid_actions:
            action_dict[action] = 0
        self.Q_Table[state] = action_dict
        self.N_Table[state] = action_dict.copy()
    
    #Q Learning via Monte Carlo method
    def act(self,state):
        state_key = tuple(state.values())
        if self.Q_Table.get(state_key) == None:
            self.generate_Q_entry(state_key)
            
        
        if self._learning:
            rolled_epsilon = random.random()
            state_epsilon = self.get_epsilon(state_key)
        
        if self._learning and rolled_epsilon < state_epsilon:
            action = self.choose_action(state_key,rand=True)
        else:
            action = self.choose_action(state_key)
            
        next_state,reward = self.environment.step(action)
        
        if next_state == None:
            self.update_Q_Table(reward)
            
        return next_state,reward
            
    def update_Q_Table(self,reward):
        for state in self.Q_Table.keys():
            for action in self.Q_Table[state].keys():
                visit_count = self.N_Table[state][action]
                current_Q = self.Q_Table[state][action]
                if visit_count != 0:
                    alpha = 1/float(visit_count)
                    #print(alpha)
                    #print(reward)
                    self.Q_Table[state][action] = current_Q + alpha*(reward - current_Q)
                

def basic_play():
    player = Basic_Player(True)
    env = Environment()
    env.add_primary_agent(player)
    env.play_game()
                
def manual_play():
    player = Manual_Player(True)
    env = Environment()
    env.add_primary_agent(player)
    env.play_game()
    
    
def naive_play():
    player = Naive_Player(True)
    env = Environment()
    env.add_primary_agent(player)
    env.play_game()
    
def qlearn_play():
    player = QLearner()
    env = Environment()
    env.add_primary_agent(player)
    env.play_game()
    
    
if __name__ == "__main__":
    qlearn_play()
    
    
    
