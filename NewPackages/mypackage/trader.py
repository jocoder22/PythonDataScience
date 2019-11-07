
class NewTrader:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __init_subclass__(cls):
        return super().__init_subclass__()

    def printall(self):
        print(self)

    