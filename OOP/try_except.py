#!/usr/bin/env python
from printdescribe import print2

class SalaryError(ValueError):
  pass

class HourError(ValueError):
  
  _message = "Hours out of range!"
  
  def __init__(self):
    ValueError.__init__(self)
    print(HourError._message)
    
    
    
    
class Worker:
  
  def __init__(self, name, wage = 15.53, hour = 8):
        self.name = name
        self.wage = wage
        
        if hour > 60:
          raise HourError
        self.hour = hour
        
        self.salary = Worker._salary()
    
    def _salary(self, bonus = 1):
        amount = self.wage * self.hour  * bonus
        # self.salary = amount
        return amount


