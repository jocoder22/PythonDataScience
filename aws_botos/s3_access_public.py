import boto3

AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"
BucketName = "bucketStaging"

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')

sp = {"sep":"\n\n", "end":"\n\n"}

# Create the boto3 client for interacting with S3 and SNS
s3 = boto3.client('s3', region_name='us-east-2', 
                        aws_access_key_id=AWS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET)


# Create the buckets
rstaging = s3.create_bucket(Bucket=BucketName)


# Upload file to the bucket
s3.upload_file(
    # Complete the filename where the file is located locally
    Filename='./jonah.csv', 
    # Set the key and bucket, default setting is private
    Key='jonah_final.csv', 
    Bucket=BucketName)


# change object acl
s3.put_object_acl(
    # Set the key and bucket, default setting is private
    Key='jonah_final.csv', 
    Bucket=BucketName,
    ACL = "public-read")


# Upload file to the bucket and modify the acl simultaneously
s3.upload_file(
    # Complete the filename where the file is located locally
    Filename='./jonah.csv', 
    # Set the key and bucket
    Key='jonah_final.csv', 
    Bucket=BucketName,
    # During upload, set ACL to public-read
    ExtraArgs = {
    'ACL': 'public-read'})


# get list of object in a bucket: returns a JSON object
# the object are contained in the Content list
bucketList = s3.list_object(
    Bucket=BucketName,
    Prefix="jonah")

urlLIst = []

# print the objects in the bucket and form a Public URL
for object in bucketList["Content"]:
    
    # grant public read access
    s3.put_object_acl(
        Bucket = BucketName,
        key = object["Key"],
        ACL = "public-read")
    
    # generate public url
    urlLIst.append(f"https://{BucketName}.s3.amazonaws.com/{object['Key']}")