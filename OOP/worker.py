#!/usr/bin/env python
import pandas as pd
from printdescribe import print2

class Worker:
    
    def __init__(self, name, wage = 15.53, hour = 8):
        self.name = name
        self.wage = wage
        self.hour = hour
        self.salary = Worker._salary(self)
    
    def _salary(self, bonus = 1):
        amount = self.wage * self.hour  * bonus
        return amount

    def __repr__(self):
        return f"Worker('{self.name}', {self.wage}, {self.hour})"


    def __str__(self):
        strr = f"""
        Worker:
            Name: {self.name}
            Hourly wage: {self.wage}
            Total time: {self.hour}hrs
            Total salary: ${self.salary}

        """
        return strr
    
  
class Staff(Worker):
    
    def __init__(self, name, wage = 22.29, hour = 7.5, compensation = 0.15):
        Worker.__init__(self, name, wage, hour)
        self.compensation = compensation
        self.salary = Staff._salary(self)
    
    
    def _salary(self, bonus=1):
        totalbenefits = bonus + self.compensation
        amt = Worker._salary(self, bonus = totalbenefits)
        return amt
        

class SeniorStaff(Staff):
    
    def __init__(self, name, wage = 22.29, hour = 7.5, compensation = 0.35):
        Staff.__init__(self, name, wage, hour)
        self.compensation = compensation
        self.salary = Staff._salary(self)
    
    
    # def _salary(self, bonus=1):
    #     totalbenefits = bonus + self.compensation
    #     amt = Worker._salary(self, bonus = totalbenefits)
    #     return amt

    # def __str__(self):
        
    #     tt = Worker.__str__(self)

    #     return tt + f"Total Compensation: {self.compensation}"
        
    #     #  """
     
class Employee:
    def __init__(self, name, salary=30000):
        self.name = name
        self.salary = salary

    def give_raise(self, amount):
        self.salary += amount

        
class Manager(Employee):
    def display(self):
        print("Manager ", self.name)

    def __init__(self, name, salary=50000, project=None):
        Employee.__init__(self, name, salary)
        self.project = project

    # Add a give_raise method
    def give_raise(self, amount, bonus=1.05):
        new_amount = amount * bonus
        Employee.give_raise(self, new_amount)

    
    def __repr__(self):
        return f"Manager('{self.name}', {self.salary}, '{self.project}')"


    def __str__(self):
        strr = f"""
        Manager:
            Name: {self.name}
            Total salary: ${self.salary}
            Projet: {self.project}
        """
        return strr
    
    
mngr = Manager("Ashta Dunbar", 78500)
mngr.give_raise(1000)
print(mngr.salary)
mngr.give_raise(2000, bonus=1.03)
print(mngr.salary)


worker1 = Worker("Amy")
staff1 = Staff("Ashley")
staff2 = Staff("Bruny", wage=20.0)
staff2_s = SeniorStaff("Senior Bruny", wage=20.0)
print2(staff1.salary, worker1.salary, staff2.salary, repr(staff2), staff2, mngr, staff2_s)
    
