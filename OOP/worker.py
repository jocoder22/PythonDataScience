#!/usr/bin/env python
import pandas as pd

class Worker:
  
  def __init__(self, name, wage = 15.53, hour = 8):
    self.name = name
    self.wage = wage
    self.hour = hour
    self.salary = Worker.salary()

  def salary(self):
    amount = self.wage * self.hour
    return amount
  
  
class Staff(Worker):
  
  def __init__(self, name, wage = 22.29, hour = 7.5, bonus = 1.05):
    Worker.__init__(self, name, wage, hour)
    self.bonus = bonus
    
    
   def get_reward(self, bonus):
      self.salary +* bonus
    
