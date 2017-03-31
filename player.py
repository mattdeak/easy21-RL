class Basic_Player:
    
    def __init__(self,environment):
        self.total = 0
        self.environment = environment
        
    def clear(self):
        self.total = 0
        self.cards = []
        
    def draw(self,suit=None):
        card = self.environment.draw(suit)
        suit,value = card.split('_')
        if suit == 'r':
            self.total -= int(value)
        else:
            self.total += int(value)
        return card
        
    def act(self,state):
        action = 'hit'
        next_state,reward = self.environment.step(action)        
        return reward
        
        
class MontePlayer(Basic_Player):
    
    def __init__(self,environment,epsilon=0.9):
        super.__init__()
        self.Q_Table = {}
    
    ##To Do
    #def act(self,state):
        
                
                
class Dealer(Basic_Player):
    
    def act(self):
        if self.total < 17:
            return "hit"
        else:
            return "stick"
            
    
