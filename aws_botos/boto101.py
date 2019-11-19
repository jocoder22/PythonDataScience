import boto3


AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"


# Create the boto3 client for interacting with S3 and SNS
s3 = boto3.client('s3', region_name='us-east-2', 
                         aws_access_key_id=AWS_KEY_ID, 
                         aws_secret_access_key=AWS_SECRET)

sns = boto3.client('sns', region_name='us-east-2', 
                         aws_access_key_id=AWS_KEY_ID, 
                         aws_secret_access_key=AWS_SECRET)

# List S3 myBuckets and SNS nyTopics
myBuckets = s3.list_buckets()
nyTopics = sns.list_topics()


# Iterate over Buckets from .list_buckets() response
for bucket in myBuckets['Buckets']:
  
  	# Print the Name for each bucket
    print(bucket['Name'])


# Create the buckets
rstaging = s3.create_bucket(Bucket='bucketStaging')
rprocessed = s3.create_bucket(Bucket='bucketProcessed')
rtest = s3.create_bucket(Bucket='bucketTest')



