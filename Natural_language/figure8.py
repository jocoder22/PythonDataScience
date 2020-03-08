import pandas as pd


comeg = "https://d1p17r2m4rzlbo.cloudfront.net/wp-content/uploads/2016/03/Corporate-messaging-DFE.csv"

message = pd.read_csv(comeg, encoding='latin-1')
print(message.columns)

message = pd.read_csv(comeg, encoding='latin-1')
mess = message[(message['category'] != "Exclude") & (message[ 'category:confidence'] == 1)]

print(mess.category.value_counts())
