class Set:
    def __init__(self, values=None):
        self.dict = {}
        if values is not None:
            for value in values:
                self.add(value)

    def __repr__(self):
        return str(self.dict)

    def add(self, value):
        self.dict[value] = value*2

    def contains(self, value):
        return value in self.dict

    def remove(self, value):
        del self.dict[value]