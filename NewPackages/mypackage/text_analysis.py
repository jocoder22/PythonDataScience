from collections import Counter
from .util import tokenize, plotcount

class TextTokenizer:
    """
    This is a class to form text tonizer

    Input: text

    """

    def __init__(self, text):
        self.greet = "This from the initiator"
        self.text = text
        self.token = self._tokenizer()
        self.wordcount = self._countwords()
    
    def _tokenizer(self):
        # tokenize text
        return tokenize(self.text)

    
    def _countwords(self):
        # count the words
        return Counter(self.token)

    def plot_count(self, attribute='wordcount', n_common=6):
        # fuction for histogram plot
        plotcount(getattr(self, attribute), n_common)



