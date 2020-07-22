#!/usr/bin/env python
from printdescribe import print2

class Worker:
    
    def __init__(self, name, newsalary):
        self.name = name
        self._salary = newsalary
        
    def __str__(self):
        str = f"""
            Worker:
                name = {self.name}
                Salary = ${self.salary} 
        """

    @property
    def salary(self):
        return self._salary
    
    @salary.setter
    def salary(self):
        if self._salary < 50000 or self._salary > 150000:
            raise ValueError("Salary out of range")
         
        self._salary = newsalary
