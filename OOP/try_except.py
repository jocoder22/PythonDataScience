#!/usr/bin/env python
from printdescribe import print2

class SalaryError(ValueError):
  pass

class HourError(ValueError):
  
  _message = "Hours out of range!"
  
  def __init__(self):
    ValueError.__init__(self)

        
  def __str__(self):
    print(HourError._message)
    return HourError._message
        
    
    
class Worker:
  
  _MAX_HOUR = 60
  
  def __init__(self, name, wage = 15.53, hour = 8):
        self.name = name
        self.wage = wage
        
        if self.hour > Worker._MAX_HOUR:
          raise HourError
          
        self.hour = hour
        
        
        self.salary = Worker._salary()
    
    def _salary(self, bonus = 1):
        amount = self.wage * self.hour  * bonus
        # self.salary = amount
        return amount


