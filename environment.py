#from player import Player
from numpy import random

class Environment:

    def __init__(self):
        
        self.player = None
        self.valid_actions = ['hit','stick']
        self.state = {}
        self.game_state = False

	#TODO: Inner class dealer
    
    def _game_start(self):
        dealer_card = self._draw('b')
        player_card = self._draw('r')
        
        dealer_value,player_value = dealer_card.split('_')[1],player_card.split('_')[1]
        self.state = {'p_sum':int(player_value),'d_start':int(dealer_value)}
        
        
    def play_game(self):
        
        self._game_start()
        reward = None
        self.game_state = True
        current_state = self.state
        status = 10000
        while self.game_state:
            next_state,reward = self.player.act(self.state)
            if next_state == "Terminal":
                if reward == 1:
                    status = 1
                if reward == -1:
                    status = 0
                else:
                    status = 0
            else:
                current_state = next_state
                    
        return current_state,reward,status
        
    def add_primary_agent(self,agent):
        self.player = agent
        agent.add_environment(self)
            

    def _draw(self,suit=None):
        number = random.choice([i for i in range(1,11)])
        
        if suit == None:
            suit = random.choice(['r','b'],p=[float(0.33333),float(0.66667)])
        
        return "{}_{}".format(suit,number)
        
    def _update_value(self,initial_value,card):
        suit,value = card.split('_')
        
        if suit == 'b':
            return initial_value + int(value)
        else:
            return initial_value - int(value)
        
    
    def step(self,action):
        reward = 0
        
        if action == 'hit':
            player_card = self._draw()
            initial_score = self.state['p_sum']
            
            new_score = self._update_value(initial_score,player_card)
            
            
                
            if new_score <= 21:
                self.state['p_sum'] = new_score
                reward = 0
            else:
                reward = -1
                self.game_state = False
            
        else:
            dealer_score = self.state['d_start']
            while  dealer_score < 17:
                new_card = self._draw()
                dealer_score = self._update_value(dealer_score,new_card)
                
            if dealer_score > 21 or self.state['p_sum'] > dealer_score:
                reward = 1
            elif dealer_score == self.state['p_sum']:
                reward = 0
            else:
                reward = -1
            
            self.game_state = False
            self.state ="Terminal"
                                
        return self.state,reward        
        
	

def _draw_test(environment,tests=30):
    for i in range(tests):
        card = environment._draw()
        print(card)

if __name__ == "__main__":
    env = Environment()
    _draw_test(env)
    