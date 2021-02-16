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

"""
And landscape object that maps all current pixobe locations.
And produces the image.
And generates food.

"""



"""
Pixobe object that governs their behaviour and evolution.

"""

class Pixobe:
    
    def __init__(self, Pixobe parent):
        # Larger sizes require more food
        self.shape = 
        # Binary desire to move forward
        self.drive = 
        # Maximum distance can move forward in one step
        self.speed = 
        # Binary desire to turn
        self.stability
        # Maximum angle can rotate in one turn
        self.spin = 
        self.fertility = 
        self.life =
        self.age = 0
    
    # You only get one attempt to step or turn per iteration
    def step(self):
        # Random choice in [0, speed]
    def turn(self):
        # Random choice in [-spin, +spin]
    
    # Triggered by touching food
    def feed(self):
    def breed(self):
        # Evolutionary improvements get exponentially harder
    def die(self):





