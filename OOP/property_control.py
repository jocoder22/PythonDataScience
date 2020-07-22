#!/usr/bin/env python
from printdescribe import print2

class Worker:
    
    def __init__(self, name, newsalary):
        self.name = name
        self._salary = newsalary

    
    def _salary(self, bonus = 1):
        amount = self.wage * self.hour  * bonus
        # self.salary = amount
        return amount
