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

# print topic Arn
health_alerts_arn = response['TopicArn']
print(health_alerts_arn)


# Create list of topic
topics = ['music', 'news', 'politics', 'health', 'vacation', 'religion']

for topic in topics:
  	# create topic for every issues
    sns.create_topic(Name=f"{topic}_generalNews")
    
    # create topic for every breaking News
    sns.create_topic(Name="{topic}_breakingNews")



