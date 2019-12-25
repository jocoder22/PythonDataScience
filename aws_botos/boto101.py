import boto3


AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
  
sp = {"sep":"\n\n", "end":"\n\n"}      

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
    print(bucket['Name'], **sp)




# Create the buckets
rstaging = s3.create_bucket(Bucket='bucketStaging')
rprocessed = s3.create_bucket(Bucket='bucketProcessed')
rtest = s3.create_bucket(Bucket='bucketTest')


# Delete the bucketStaging bucket
s3.delete_bucket(Bucket='bucketStaging')

# Get the names of buckets
bucketList = s3.list_buckets()

# Print each Buckets Name
for bucket in bucketList['Buckets']:
    print(bucket['Name'], **sp)
    
    
# Delete all the buckets with 'bucketStaging'
for bucket in bucketList['Buckets']:
  if 'bucketStaging' in bucket['Name']:
      s3.delete_bucket(Bucket=bucket['Name'])
    


