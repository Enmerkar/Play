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
        self.shape = 
        self.speed = 
        self.spin = 
        self.fertility = 
        self.life =
        self.age = 0
    
    # You only get one attempt to step or turn per iteration
    def step(self):
    def turn(self):
    
    # Triggered by touching food
    def feed(self):
    def breed(self):
    def die(self):

