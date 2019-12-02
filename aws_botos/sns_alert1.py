import boto3

AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"
BucketName = "htmlObjects"

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
        
sp = {"sep":"\n\n", "end":"\n\n"}

# Create the boto3 client for interacting SNS
sns = boto3.client('sns', region_name='us-east-2', 
                        aws_access_key_id=AWS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET)

# Create the health_alerts topic
response = sns.create_topic(Name="health_alerts")
respArn = sns.create_topic(Name="health_alerts")['TopicArn']

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
     
print(sns.list_topics()['Topics'])


# subscribe to individual to topic, using phone number and email
phoneNumbers = ["+12128953490", "+1262863490", "+14188953490", "+13128953490"]
emails = ["butks@gmail.com", "monmm@yahoo.com", "yonaf@soft.com", "honye@gmail.com"]


# subscribe members to all topics
for topic in topicList:
      # subscribe to sms
    for i, phoneNumber in enumerate(phoneNumber):
        sns.subscribe(TopicArn = topic['TopicArn'],
            Protocol = "sms", # check for list ["email, sns"]
            Endpoint = phoneNumber ) # check Endpoint = [emails[i], phonenumber] 
      
      # subscribe to email
        sns.subscribe(TopicArn = topic['TopicArn'],
                  Protocol = "email", # check for list ["email, sns"]
                  Endpoint = emails[i]) # check Endpoint = [emails[i], phonenumber]


# List subscriptions for each topic.
for topic in topicList:
      if "generalNews" in topic['TopicArn']:
            response = sns.list_subscriptions_by_topic(TopicArn = topic['TopicArn'])

# For each subscription, if the protocol is SMS, unsubscribe
for subject in response['Subscriptions']:
  if subject['Protocol'] == 'sms':
	  sns.unsubscribe(SubscriptionArn=subject['SubscriptionArn'])

# List subscriptions for health alerts topic in one line
alerts = sns.list_subscriptions_by_topic(
  TopicArn=health_alerts_arn)['Subscriptions']

# Print the subscriptions
print(alerts)