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
respArn = sns.create_topic(Name="health_alerts")['TopicArn']

# send single sms
sns.publish(PhoneNumber="+13234569086",
            Message="Hello World")

# send to multiple subscriber
temperature = 48


if temperature <= 25:
    # # The message should contain the number of temperature.
    message = f"The temperature is {temperature}F, please prepare to take action!"
    
    # The email subject should also contain the temperature
    subject = f"Latest is {temperature}"

    # Publish the email to the Health Alert topic
    sns.publish(
        TopicArn = respArn,
        # Set subject and message
        Message = message,
        Subject = subject)
    