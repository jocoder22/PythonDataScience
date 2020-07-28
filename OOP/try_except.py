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
  _MAX_SALARY = 150000

  
  def __init__(self, name, wage = 15.53, hour = 8, bonus=1):
    self.name = name
    self.wage = wage
    
    if hour > Worker._MAX_HOUR:
      raise HourError()
    
    self.hour = hour
    self.bonus = bonus
    self._salary = 0

  def salary_cal(self):
    _amount_sal = self.wage * self.hour  * self.bonus
  
    if _amount_sal > Worker._MAX_SALARY:
      raise SalaryError("Salary out of range!")
    return _amount_sal

  @property
  def salary(self):
    self._salary = Worker.salary_cal(self)
    return self._salary

  def __repr__(self):
    return f"Worker('{self.name}', {self.wage}, {self.hour}, {self.bonus})"

  def __str__(self):
    strr = f"""
      Worker:
        Name: {self.name}
        Hourly wages: ${self.wage}
        Hours  worked: {self.hour}
        Bonus earned: {self.bonus}
        Total salary: ${self.salary}
    """
    return strr

john = Worker("John Smith", 43.50, 50)
jane = Worker("Jane Pretty", 196.00, 40)
print2(jane, repr(jane))