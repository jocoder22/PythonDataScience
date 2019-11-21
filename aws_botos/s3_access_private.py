import boto3

AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"
BucketName = "bucketStaging"

# Create the boto3 client for interacting with S3 and SNS
s3 = boto3.client('s3', region_name='us-east-2', 
                        aws_access_key_id=AWS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET)


# Download file to the bucket
s3.download_file(
    # Complete the filename where the file will be stored locally
    Filename='./jonah.csv', 
    # Set the key and bucket, default setting is private
    Key='jonah_final.csv', 
    Bucket=BucketName)
