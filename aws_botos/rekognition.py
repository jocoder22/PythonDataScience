import boto3

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')

AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"
BucketName = "ImageBucket"


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
print(image2recog['Labels'])
      
      
# Use Rekognition client to detect images
image1recogBucket = recog.detect_labels(
    Image={"S3Object":{
        "Bucket": BucketName,
        "Name": "image1.jpg"
            }
        }, MaxLabels=6,
           MinConfidence=95)

# Print the labels
print(image1recogBucket['Labels'])



###############################################################################
######################## Detect words #########################################
###############################################################################

# Use Rekognition client to detect text in images
textrecog = recog.detect_text(
    Image={"S3Object":{
        "Bucket": BucketName,
        "Name": "image1.jpg"
            }
        }, MaxLabels=6,
           MinConfidence=95)

# Create empty list of words
words = []

# Iterate over the TextDetections in the textrecog dictionary
for text in textrecog['TextDetections']:
  	# If TextDetection type is WORD, append it to words list
    if text['Type'] == 'WORD':  # other type == LINE
        # Append the detected text
        words.append(text['DetectedText'])
        
        
# Print out the words list
print(words)
