# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:32:35 2021

@author: Justin

A 2D environment with pixel lifeforms. They can reach rotate and translate. They die after a time. They reproduce. They eat randomly generated food. Offspring can differ by:
1. Shape (add/remove one pixel)
2. Rotation behaviour
3. Translation behaviour
4. Reproduction behaviour
5. Age of death

Iterate through each being, first come first served.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
And landscape object that maps all current pixobe locations.
And produces the image.
And generates food.

"""

class Land:
    
    def __init__(self, height = 30, width = 30):
        self.height = height
        self.width = width
        self.grid = np.zeros((self.height, self.width))
        
    def draw(self):
        plt.figure()
        sns.heatmap(self.grid,
                    square=True,
                    xticklabels=False,
                    yticklabels=False,
                    cbar=False)
        # IDEA: hexbin


"""
Pixobe object that governs their behaviour and evolution.

"""

class Pixobe:
    
    def __init__(self, land):
        self.land = land
        # Position of pixobe's head in environment
        self.location = [5,5]
        # Age in iterations lived
        self.age = 0
        # Steps can take per cycle
        self.speed = 1
    
    # You only get one attempt to step or turn per iteration
    def step(self):
        self.location[0] += self.speed
    
    # Redraw pixobe in Environment
    def move(self):
        self.land.grid[self.location[0], self.location[1]] = 1
        


"""
World controller

"""
  
class World:
    
    def __init__(self, pixobes = 5):
        self.land = Land(50, 50)
        self.census = pd.DataFrame({'Label': pd.Series([], dtype='int')
                                    ,'Pixobe': pd.Series([], dtype='object')})
        for i in range(1, pixobes+1):
            self.census = self.census.append({'Label':i, 'Pixobe':Pixobe(self.land)}, ignore_index=True)
        



"""
Run test

"""

W = World()
vars(W)

E = Environment()
P = Pixobe(E)
E.grid
E.draw()
P.step()
P.move()
E.draw()

P.step()
P.move()
E.draw()
P.step()
P.move()
E.draw()



"""
Pixobe development/idea area

"""       

class PixobeIdeas:
    
    def __init__(self, Pixobe parent):
        # Position of pixobe's head in environment
        self.location = [5,5]
        # Direction head is facing. 
        self.orientation = 0
        self.shape = 1
        self.speed = 1
        self.spin = 1
        # Minimum cycles between reproductions
        self.fertility = 100
        # Maximum cycles pixobe can live
        self.life = 1000
        self.age = 0
    
    # You only get one attempt to step or turn per iteration
    def step(self):
    def turn(self):
    
    # Triggered by touching food
    def feed(self):
    def breed(self):
    def die(self):

