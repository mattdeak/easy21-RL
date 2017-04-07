"""This module contains actors for the environment.
Actors contained in this module are naive and cannot learn.
"""
__author__ = "Matthew Deakos"

from numpy import random

class Basic_Player:
    """The base class for players integrating with an environment.
    """
    
    def __init__(self,debug=False):
        self.environment = None
        self._debug = False
        
    def act(self,state):
        if self.environment == None:
            raise ValueError("Must add an environment in order to act")
        
        #Take an action randomly
        action = self.choose_action(state)
        next_state,reward = self.environment.step(action)        
        return next_state,reward
        
    def choose_action(self,state):
        """Choose an action randomly in the environment.
        """        
        action = random.choice(self.environment.valid_actions)
        
        if self._debug:
            print(state['p_sum'],action)
        return action
        

        
class Naive_Player(Basic_Player):
    
    def choose_action(self,state):
        if state['p_sum'] >= 16:
            action = 'stick'
        else:
            action = 'hit'
            
        if self._debug:
            print(state['p_sum'],action)
            
        return action
            
         
class Manual_Player(Basic_Player):
    
    def choose_action(self,state):
        print("Current State: n {}".format(state))
        action = None
        while action == None:
            print("Valid Actions are: ")
            print(",".join(self.environment.valid_actions))
            choice = input("Choose your action: ")
            if choice in self.environment.valid_actions:
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
    
    
    
