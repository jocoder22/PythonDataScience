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
response = sns.create_topic(Name="health_alerts")['TopicArn']

# print topic Arn, Amazon Resource Name
health_alerts_arn = response['TopicArn']
print(health_alerts_arn)


# Create list of topic
topics = ['music', 'news', 'politics', 'health', 'vacation', 'religion']


for topic in topics:
  	# create topic for every issues
    sns.create_topic(Name=f"{topic}_generalNews")
    
    # create topic for every breaking News
    sns.create_topic(Name="{topic}_breakingNews")


# Print all the topics in SNS
response = sns.list_topics()
print(response['Topics'])

# Delete Topics
# Delete one Topic
sns.delete_topic(TopicArn = health_alerts_arn)


# Delete multiple Topics, use looping on list of topics
# first get the list of topics
topicList = sns.list_topics()['Topic']

# Iterate over the topicList and delete selected topic using TopicArn
for topic in topicList:
   if "generalNews" in topic['TopicArn']:
     sns.delete_topic(TopicArn = topic['TopicArn'])
     
