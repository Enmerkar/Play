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
from random import randrange

"""
And landscape object that maps all current pixobe locations.
And produces the image.
And generates food.

"""

class Land:
    
    def __init__(self, height=30, width=30):
        self.height = height
        self.width = width
        self.grid = np.zeros((self.height, self.width))
    
    def get_height(self):
        return self.height
    
    def get_width(self):
        return self.width
    
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
    
    def __init__(self, land, location=False):
        self.land = land
        # Position of pixobe's head in environment
        if location:
            #TODO: place near parent
        else:
            self.location = [randrange(0,land.get_height()),
                             randrange(0,land.get_width())]
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
  
def run(pixobes = 5, iterations = 100):
    land = Land(50,50)
    census = pd.DataFrame({'Label': pd.Series([], dtype='int')
                                ,'Pixobe': pd.Series([], dtype='object')})
    for i in range(1, pixobes+1):
        census = census.append({'Label':i,
                                'Pixobe':Pixobe(land, location=True)},
                               ignore_index=True)
    
    land.draw()

# Execute

run(10)


