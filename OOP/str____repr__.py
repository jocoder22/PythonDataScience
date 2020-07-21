#!/usr/bin/env python
from printdescribe import print2

class Patients:

    def __init__(self, name, id, gender):
        self.name = name
        self.id = id
        self.gender = gender

    def __str__(self):

        patient_str = f"""
        Patient:
            Name : {self.name}
            ID : {self.id}
            Gender : {self.gender}
        """

        return patient_str

class Staff:

    def __init__(self, name, id, gender):
        self.name = name
        self.id = id
        self.gender = gender


    def __str__(self):

        staff_str = f"""
        Patient:
            Name : {self.name}
            ID : {self.id}
            Gender : {self.gender}
        """

        return staff_str
    
    
    def __repr__(self):

        staff_repr = f"Staff('{self.name}', {self.id}, '{self.gender}')"

        return staff_repr

    

patient1 = Patients("Charles", 459234, "Male")
str(patient1)
patient5 = Staff("Charles", 894863, "Male")

print2(patient1, patient5, repr(patient5))
str(patient5)