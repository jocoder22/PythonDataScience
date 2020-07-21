#!/usr/bin/env python
from printdescribe import print2

class Patients:

    def __init__(self, name, id, gender):
        self.name = name
        self.id = id
        self.gender = gender

    def __eq__(self, other):
        return self.name == other.name and self.id == other.id\
            and  type(self) == type(other)

class Staff:

    
    def __init__(self, name, id, gender):
        self.name = name
        self.id = id
        self.gender = gender


patient1 = Patients("Charles", 459234, "Male")
patient2 = Patients("Charles", 4593234, "Male")
patient3 = Patients("Marylene", 459234, "Female")
patient4 = Patients("Charles", 459234, "Male")
patient4 = Staff("Charles", 459234, "Male")

print2(patient1 == patient2, patient3 == patient1, 
    patient1 == patient4, patient4 == patient1)