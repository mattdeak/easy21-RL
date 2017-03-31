
from numpy import random
from collections import defaultdict

class Basic_Player:
    
    def __init__(self):
        self.environment = None
        self.valid_actions = None
        
    def add_environment(self,environment):
        self.environment = environment
        self.valid_actions = environment.valid_actions
        
    def act(self,state):
        if self.environment == None:
            raise ValueError("Must add an environment in order to act")
        
        #Choose an action randomly
        action = random.choice(self.valid_actions)
        
        next_state,reward = self.environment.step(action)        
        return reward
        
        
class QLearner_Monte(Basic_Player):
    
    def __init__(self,epsilon=0.9):
        super.__init__()
        self.Q_Table = {}
    
    ##To Do
    #def act(self,state):
        
                
    
            
    
