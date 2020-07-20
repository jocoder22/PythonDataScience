#!/usr/bin/env python
import pandas as pd

class Worker:
  
  def __init__(self, name, wage = 15.53, hour = 8):
    self.name = name
    self.wage = wage
    self.hour = hour
#     self.salary = Worker.salary()
    self.salary = self.wage * self.hour

  def newsalary(self, bonus = 1):
    amount = self.wage * self.hour  * bonus
    self.salary = amount
  
  
class Staff(Worker):
  
  def __init__(self, name, wage = 22.29, hour = 7.5, compensation = 0.15):
    Worker.__init__(self, name, wage, hour)
    self.compensation = compensation
    
    
   def give_reward(self):
      Worker.newsalary(bonus = self.compensation)
      
   def give_raise(self):
      Worker.newsalary(bonus = 1.057)
      
     
    
