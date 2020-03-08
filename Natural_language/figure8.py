import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet'])


import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nlkt.corpus import stopwords



def load_data():
  comeg = "https://d1p17r2m4rzlbo.cloudfront.net/wp-content/uploads/2016/03/Corporate-messaging-DFE.csv"

  message = pd.read_csv(comeg, encoding='latin-1')
#   print(message.columns)

  mess = message[(message['category'] != "Exclude") & (message[ 'category:confidence'] == 1)]
  
  x = message.text.values()
  y = message.category.values()
  
  return x, y


