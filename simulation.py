# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:40:16 2017

@author: matthew
"""
    
    
def simulate(player,environment,n_trials=1000):
    
    
    environment.add_primary_agent(player)
    wins = []
    
    print("Simulating...")
    for i in range(n_trials):
        if i % n_trials/10 == 0:
            print ("Loading game {}".format(i))
        _,_,result = environment.play_game()
        wins.append(result)
        
    win_rate = float(sum(wins))/n_trials
        
    return wins,win_rate

        
        
