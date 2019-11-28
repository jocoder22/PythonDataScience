import boto3

AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"
BucketName = "htmlObjects"


# Create the boto3 client for interacting SNS
recog = boto3.client('translate', region_name='us-east-2', 
                        aws_access_key_id=AWS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET)