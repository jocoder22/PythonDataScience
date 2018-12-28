#!/usr/bin/env python
class Set:
    def __init__(self, values=None):
        self.dict = {}
        if values is not None:
            for value in values:
                self.add(value)

    def __repr__(self):
        return str(self.dict)

    def add(self, value):
        self.dict[value] = value ** 2

    def contains(self, value):
        return value in self.dict

    def remove(self, value):
        del self.dict[value]


xt = Set([8, 3, 6, 11])
print(xt)
xt.add(10)
xt.contains(23)
xt.remove(6)
xt.contains(6)
