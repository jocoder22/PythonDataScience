import pandas as pd
import boto3

AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"
BucketName = "bucketStaging"

# Create the boto3 client for interacting with S3 and SNS
s3 = boto3.client('s3', region_name='us-east-2', 
                        aws_access_key_id=AWS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET)


# Download file to the bucket
s3.download_file(
    # Complete the filename where the file will be stored locally
    Filename='./jonah.csv', 
    # Set the key and bucket, default setting is private
    Key='jonah_final.csv', 
    Bucket=BucketName)

# convert to pandas dataframe
df = pd.read_csv("./jonah.csv")


# using get_object(). return JSON with streaming Body 
response = s3.get_object(Key='jonah_final.csv', Bucket=BucketName)

# convert to pandas dataframe
df2 = pd.read_csv(response['Body'])


###################################################################
#################### using presigned URLs #########################
# to grant temporary access, that expires after predefined time ###
# you can shared this generated presigned url with other coworker #
###################################################################
sharedURL = s3.generate_presigned_url(
    ClientMethod = "get_object",
    ExpiresIn = 3600,
    params = {"Key": "jonah_final.csv", 
              "Bucket": BucketName})

df3 = pd.read_csv(sharedURL)


# downloading multiple file with same structure
# first get the list of files
df4 = []
fileList = s3.list_object(Bucket=BucketName, Prefix="Jan")["Content"]

# loop over the file list
for file in fileList:
    # download the file
    s3.get_object(Bucket=BucketName, Key=file['Key'])