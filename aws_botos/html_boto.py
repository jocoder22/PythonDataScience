import pandas as pd 
import boto3

import os

print(os.getcwd())
path = r"D:\PythonDataScience\aws_botos\html"
os.chdir(path)

with open("D:\TimerSeriesAnalysis\AMZN.csv") as file:
    df = pd.read_csv(file)

print(df.head())

# Generate an HTML table with no border and selected columns
df.to_html('./people_No_border.html', render_links=True,
           # Keep specific columns only
           columns= "Date,Open,Close,Adj Close,Volume".split(","), 
           # Set border
           border=0)

# Generate an html table with border and all columns.
df.to_html('./people_border.html',render_links=True,
           border=1)


##################################################################################
############################### Create bucket ####################################
###################################################################################

AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"
BucketName = "htmlObjects"

# Create the boto3 client for interacting with S3 and SNS
s3 = boto3.client('s3', region_name='us-east-2', 
                        aws_access_key_id=AWS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET)


# Upload the lines.html file to S3
filelist = ["people_border.html", "people_No_border.html.html"]
keylist = ["s3people_border.html", "s3people_No_border.html.html"]
publicUrlList = []

for i, file in enumerate(filelist):
    s3.upload_file(Filename=file, 
               # Set the bucket name
               Bucket='htmlObjects', Key=keylist[i],
               # Configure uploaded file
               ExtraArgs = {
                 # Set proper content type
                 'ContentType':'text/html',
                 # Set proper ACL
                 'ACL': 'public-read'})
    
    # generate public url
    publicUrl = f"http://htmlObjects.s3.amazonaws.com/{keylist[i]}"
    publicUrlList.append(publicUrl)
    print(publicUrl)
    
    # add dataset to df
    if i > 0:
        df.loc[i, ["links"]] = publicUrl
    else:
        df["links"] = publicUrl

# Generate an html table with border and all columns.
df.to_html('./people_border.html',render_links=True,
           border=1)


