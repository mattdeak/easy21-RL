#from player import Player
from numpy import random
import player
class Environment:

    def __init__(self):
        
        self.player = None
        self.valid_actions = ['hit','stick']
        self.state = {}
        self.game_state = False

	#TODO: Inner class dealer
    
    def game_start(self):
        dealer_card = self.draw('b')
        player_card = self.draw('r')
        
        dealer_value,player_value = dealer_card.split('_')[1],player_card.split('_')[1]
        self.state = {'p_sum':int(player_value),'d_start':int(dealer_value)}
        
        
    def play_game(self):
        
        self.game_start()
        reward = None
        self.game_state = True
        while self.game_state:
            reward = self.player.act(self.state)
            print(reward)
        return self.state,reward
        
    def add_primary_agent(self,agent):
        self.player = agent
        agent.add_environment(self)
            

    def draw(self,suit=None):
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
            player_card = self.draw()
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
                new_card = self.draw()
                dealer_score = self._update_value(dealer_score,new_card)
                
            if dealer_score > 21 or self.state['p_sum'] > dealer_score:
                reward = 1
            elif dealer_score == self.state['p_sum']:
                reward = 0
            else:
                reward = -1
            
            self.game_state = False
                                
        return self.state,reward        
        
    def reset(self):
        self.__init__()
	

def draw_test(environment,tests=30):
    for i in range(tests):
        card = environment.draw()
        print(card)

if __name__ == "__main__":
    env = Environment()
    draw_test(env)
    