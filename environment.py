#from player import Player
from numpy import random
import player
class Environment:

    def __init__(self,player_type='Basic'):
        self.dealer = player.Dealer(self)
        if player_type == 'Basic':
            self.player = player.Basic_Player(self)
        self.state = {}
        self.game_state = False

	#TODO: Inner class dealer
    
    def game_start(self):
        self.dealer.draw('b')
        self.player.draw('r')
        self.state = {'p_sum':self.player.total,'d_start':self.dealer.total}
        
        
    def play_game(self):
        
        self.reset()
        self.game_start()
        reward = None
        self.game_state = True
        while self.game_state:
            reward = self.player.act(self.state)
            print(reward)
        return self.state,reward
            
        
        
            

    def draw(self,suit=None):
        number = random.choice([i for i in range(1,11)])
        
        if suit == None:
            suit = random.choice(['r','b'],p=[float(1/3),float(2/3)])
        
        return "{}_{}".format(suit,number)
        
    
    def step(self,action):
        reward = 0
        terminal = False
        
        if action == 'hit':
            self.player.draw()
            if self.player.total <= 21:
                self.state['p_sum'] = self.player.total
                reward = 0
            else:
                reward = -1
                self.game_state = False
            
        else:
            d_action = self.dealer.act(self.state)
            while d_action != 'stick' and self.dealer.total <= 21:
                self.dealer.draw()
                
            if self.dealer.total > 21 or self.player.total > self.dealer.total:
                reward = 1
            elif self.dealer.total == self.player.total:
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
    