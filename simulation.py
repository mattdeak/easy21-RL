# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:40:16 2017

@author: matthew
"""
from environment import Easy21_Environment
from q_players import QLearner
import traceback as tb
import sys
import matplotlib.pyplot as plt
import numpy as np
    
def simulate(player,environment,n_trials=100000):
    
    
    environment.add_primary_agent(player)
    wins = []
    
    print("Simulating...")
    for i in range(n_trials):
        if i % (n_trials/10) == 0:
            print ("Loading game {}".format(i))
        try:
            _,_,result = environment.play_game()
            wins.append(result)
        except Exception:
            tb.print_exc(file=sys.stdout)
        
        
    win_rate = float(sum(wins))/n_trials
        
    return wins,win_rate
    
if __name__ == "__main__":
    env = Easy21_Environment()
    p = QLearner()
    wins,win_rate = simulate(p,envn_trials=1000000)
    print("Win Rate: {}".format(win_rate))
    plt.plot(range(len(wins)),np.cumsum(wins))
    

        
        
