from numpy import random

# This is the abstract-like parent class from which all prey types get their structure

class Animal:
    win_val = 10
    lose_val = 1

    @classmethod
    def win(self):
        return __class__.win_val 
    @classmethod
    def lose(self):
        return __class__.lose_val
    
# Stag: 
#       Agree: 10, 10
#       Disagree: 1, 8  

class Stag(Animal):
    pass

# Hare: 
#       Agree: 5, 5
#       Disagree: 8, 1  

class Hare(Animal):
    win_val = 5
    lose_val = 8
    @classmethod
    def win(self):
        return __class__.win_val 
    @classmethod
    def lose(self):
        return __class__.lose_val
# Bison: 
#       Agree: 15 / (1.001 * number of past agreements), 15 / (1.001 * number of past agreements)
#       Disagree: 1, 8      

class Bison(Animal):

    win_val = 15
    lose_val = 1
    decline_rate = 1.0001
    @classmethod
    def win(self):
        __class__.win_val = max(__class__.win_val / __class__.decline_rate, 0)
        return __class__.win_val
    
# Random: 
#       Agree: Random integer [6, 15), Random integer [6, 15)
#       Disagree: 1, 8

class Random(Animal):
    @classmethod
    def win(self):
        return random.randint(6,15)  