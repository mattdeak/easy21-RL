"""This module contains actors for the environment.
Actors contained in this module are naive and cannot learn.
"""
__author__ = "Matthew Deakos"

from numpy import random

class Basic_Player:
    """The base class for players integrating with an environment.
    """

    def __init__(self):
        """Initializer for the base interactive agent.
        
        Attributes:
            environment: The environment that this class is acting on. 
        """
        self.environment = None
        
    def act(self,state):
        """Takes a random action in the environment."""
        if self.environment == None:
            raise ValueError("Must add an environment in order to act")
        
        #Take an action randomly
        action = self.choose_action(state)
        next_state,reward = self.environment.step(action)        
        return next_state,reward
        
    def choose_action(self,state):
        """Choose an action randomly in the environment."""
        action = random.choice(self.environment.valid_actions)
        
        return action
        
   
class Naive_Player(Basic_Player):
    """The naive player acts in the environment according to a naive,
    hardcoded policy.
    """
    
    def choose_action(self,state):
        """Choose an action based on the naive policy:
        If the current sum is greater than 16, stick.
        Otherwise, hit.
        """
        if state['p_sum'] >= 16:
            action = 'stick'
        else:
            action = 'hit'
            
        return action
            
         
class Manual_Player(Basic_Player):
    """Lets the user decide what actions to take in the environment."""
    
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
        
    def act(self,state):
        """Takes an action and prints the reward of that action."""
        r = super().act(state)
        print("Reward: {}".format(r))
        return r
    
    
    
