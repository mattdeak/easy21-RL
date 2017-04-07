from environment import Easy21_Environment
from numpy import random
import numpy as np
from collections import defaultdict

class Basic_Player:
    """The base class for players integrating with an environment.
    """
    
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
        return next_state,reward
        
class Naive_Player(Basic_Player):
    
    #Override act
    def choose_action(self,state):
        if state['p_sum'] >= 16:
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
    for i in range(1):
        env.play_game()
    
    
if __name__ == "__main__":
    qlearn_play()
    
    
    
