"""This module contains the Easy21 Gaming environment
"""

__author__ = "Matthew Deakos"

from numpy import random
from player import Basic_Player

class Easy21_Environment:
    """Game environment class for Easy21
    """

    def __init__(self):
        """
        Attributes:
            self.player: The playing agent
            self.valid_actions: A list of actions in the game state
            self.state: The current game state
        """
        self.valid_actions = ['hit','stick']
        self.rewards = {'win':1,'draw':0,'loss':-1}
        self.terminal_reward = None
        self.state = {}
        self._player = None
        
    @property    
    def player(self):
        """Returns the player
        """
        return self._player
        
    @player.setter
    def player(self,agent):
        """Sets the player adds this instance to the agent environment
        """
        agent.environment = self
        self._player = agent

    def play_game(self):
        """Simulate one game
        """
        #Ensure that a player exists
        assert self.player is not None, "No player given to environment"
        
        self._initialize_game()
        
        while self.state != None: 
            self.player.act(self.state)
                    
        return self.terminal_reward
    
    def step(self,action):
        """Performs an action in the environment and updates accordingly.
        """
        assert action in self.valid_actions, "{} is an invalid action.".format(action)
        
        reward = 0 #Default reward
        
        if self.state == None:
            raise ValueError("Trying to act on a terminal state.")
        
        if action == 'hit':
            player_card = self._draw() #Draw at least one new card
            initial_score = self.state['p_sum']
            new_score = self._update_value(initial_score,player_card)
            self.state['p_sum'] = new_score
            
            while self.state['p_sum'] < 11: #Auto-hit for the player if the sum is under 11
                player_card = self._draw()
                initial_score = self.state['p_sum']
                new_score = self._update_value(initial_score,player_card)
                self.state['p_sum'] = new_score
    
            if new_score <= 21:
                self.state['p_sum'] = new_score
            else:
                reward = self.rewards['loss']
                self.state = None
            
        else:
            dealer_score = self.state['d_start']
            
            while  dealer_score < 17:
                new_card = self._draw()
                dealer_score = self._update_value(dealer_score,new_card)
                
            if dealer_score > 21 or self.state['p_sum'] > dealer_score:
                reward = self.rewards['win']
            elif dealer_score == self.state['p_sum']:
                reward = self.rewards['draw']
            else:
                reward = self.rewards['loss']
            
            self.state = None
            
        if self.state == None:
            self.terminal_reward = reward
                                
        return self.state,reward   
        
    def _initialize_game(self):
        """Initializes a game state
        """
        #Set terminal reward to None
        self.terminal_reward = None
        
        #Draw two black cards
        dealer_card = self._draw('b')
        player_card = self._draw('b')
        
        #Get the card values
        dealer_value,player_value = dealer_card.split('_')[1],player_card.split('_')[1]
        
        #Set the initial state
        self.state = {'p_sum':int(player_value),'d_start':int(dealer_value)}
        
        #Keep drawing until the player value is 11 or greater
        while self.state['p_sum'] < 11:
            player_card = self._draw('b')
            self.state['p_sum'] = self._update_value(self.state['p_sum'],player_card)
            

    def _draw(self,suit=None):
        """Draws a card from the infinite deck. Weights 66% black, 33% red.
        """
        number = random.choice([i for i in range(1,11)])
        
        if suit == None:
            suit = random.choice(['r','b'],p=[float(0.33333),float(0.66667)])
        
        return "{}_{}".format(suit,number)
        
    def _update_value(self,initial_value,card): 
        """Returns an updated value
        """               
        suit,value = card.split('_')
        
        if suit == 'b':
            return initial_value + int(value)
        else:
            return initial_value - int(value)
        
        
def _draw_test(environment,tests=30):
    for i in range(tests):
        card = environment._draw()
        print(card)
        
def play_game_test():
    env = Easy21_Environment()
    player = Basic_Player()
    env.player = player
    env.play_game()
    

if __name__ == "__main__":
    play_game_test()
    