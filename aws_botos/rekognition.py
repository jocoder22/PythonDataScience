import boto3

AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"
BucketName = "htmlObjects"


# Create the boto3 client for interacting SNS
recog = boto3.client('rekognition', region_name='us-east-2', 
                        aws_access_key_id=AWS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET)


myImages = ["image1", "image2"]

# Use Rekognition client to detect images
image1recog = recog.detect_labels(
    Image=myImages[0], MaxLabels=1)

# Print the labels
print(image1recog['Labels'])


# Use Rekognition client to detect labels
image2recog = recog.detect_labels(
    Image=myImages[1], MaxLabels=1)


# Print the labels
print(image2recog = recog.detect_labels(['Labels']))
      
      
# Use Rekognition client to detect images
image1recogBucket = recog.detect_labels(
    Image={"S3Object":{
        "Bucket": "ImageBucket",
        "Name": "image1.jpg"
    }}, MaxLabels=6,
           MinConfidence=95)

# Print the labels
print(image1recogBucket['Labels'])