import pandas as pd


comeg = "https://d1p17r2m4rzlbo.cloudfront.net/wp-content/uploads/2016/03/Corporate-messaging-DFE.csv"

message = pd.read_csv(comeg, encoding='latin-1')
print(message.columns)
