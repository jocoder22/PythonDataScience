import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])


import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



def load_data():
  comeg = "https://d1p17r2m4rzlbo.cloudfront.net/wp-content/uploads/2016/03/Corporate-messaging-DFE.csv"

  message = pd.read_csv(comeg, encoding='latin-1')
#   print(message.columns)

  mess = message[(message['category'] != "Exclude") & (message[ 'category:confidence'] == 1)]
  
  x = message.text.values
  y = message.category.values
  
  return x, y


regfind = "http:.*"

def tokenize(text):
    # urlmatch = re.findall(regfind, text)
    
    # for url in urlmatch:
    #     text = text.replace(url, " ")

    regfind = r"http:.*"
    text = re.sub(regfind, " ", text)

    stop_words = set(stopwords.words('english'))
    stop_words.add(":")
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()

    clean_words = []
    
    for token in tokens:
        _token = lemmatizer.lemmatize(token).lower().strip()
        _token = lemmatizer.lemmatize(_token, pos="v")
        clean_words.append(_token)

    return clean_words


X, y = load_data()
for message in X[:5]:
    tokens = tokenize(message)
    print(message)
    print(tokens, '\n')
