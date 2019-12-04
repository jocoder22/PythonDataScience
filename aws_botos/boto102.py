import boto3


AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
        
             
# Create the boto3 client for interacting with S3 and SNS
s3 = boto3.client('s3', region_name='us-east-2', 
                         aws_access_key_id=AWS_KEY_ID, 
                         aws_secret_access_key=AWS_SECRET)



 s3.upload_file(Filename="image1.jpg", 
            # Set the bucket name
            Bucket='htmlObjects', Key="image1.jpg",
            # Configure uploaded file
            ExtraArgs = {
                # Set proper content type
                'ContentType':'jpg',
                # Set proper ACL
                'ACL': 'public-read'})   