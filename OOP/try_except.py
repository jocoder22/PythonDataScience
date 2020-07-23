#!/usr/bin/env python
from printdescribe import 

class SalaryError(ValueError):
  pass

class HourError(ValueError):
  
  _message = "Hours out of range!"
  
  def __init__(self):
    ValueError.__init__(self)
    print(HourError._message)


