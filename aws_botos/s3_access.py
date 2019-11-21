import boto3

AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"

# Create the boto3 client for interacting with S3 and SNS
s3 = boto3.client('s3', region_name='us-east-2', 
                         aws_access_key_id=AWS_KEY_ID, 
                         aws_secret_access_key=AWS_SECRET)


# Create the buckets
rstaging = s3.create_bucket(Bucket='bucketStaging')

# Upload file to the bucket
s3.upload_file(
    # Complete the filename where the file is located locally
    Filename='./jonah.csv', 
    # Set the key and bucket, default setting is private
    Key='jonah_final.csv', 
    Bucket='bucketStaging' )





# Upload file to the bucket
s3.upload_file(
    # Complete the filename where the file is located locally
    Filename='./jonah.csv', 
    # Set the key and bucket
    Key='jonah_final.csv', 
    Bucket='bucketStaging',
    # During upload, set ACL to public-read
    ExtraArgs = {
    'ACL': 'public-read'})

