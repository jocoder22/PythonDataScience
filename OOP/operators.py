#!/usr/bin/env python
from printdescribe import print2

class Patients:

    def __init__(self, name, id, gender):
        self.name = name
        self.id = id
        self.gender = gender

    def __eq__(self, other):
        # return self.name == other.name and self.id == other.id\
        #     and  type(self) == type(other)
        return self.name == other.name and self.id == other.id\
            and isinstance(other, Patients)


class Staff:

    def __init__(self, name, id, gender):
        self.name = name
        self.id = id
        self.gender = gender

    def __eq__(self, other):
        return self.name == other.name and self.id == other.id\
            and isinstance(other, Staff)



patient1 = Patients("Charles", 459234, "Male")
patient2 = Patients("Charles", 876323, "Male")
patient3 = Patients("Marylene", 459234, "Female")
patient4 = Patients("Charles", 459234, "Male")
patient5 = Staff("Charles", 459234, "Male")

print2(patient1 == patient2, patient3 == patient1, 
    patient1 == patient4)
print2("$"*20)
print2(patient1 == patient5, patient5 == patient1)