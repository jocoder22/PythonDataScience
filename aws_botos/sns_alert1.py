import boto3

AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"
BucketName = "htmlObjects"

# Create the boto3 client for interacting SNS
sns = boto3.client('sns', region_name='us-east-2', 
                        aws_access_key_id=AWS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET)

# Create the health_alerts topic
response = sns.create_topic(Name="health_alerts")
health_alerts_arn = response['TopicArn']