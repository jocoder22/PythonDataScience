import boto3

AWS_KEY_ID = "keep off this"
AWS_SECRET = "never used this"
BucketName = "htmlObjects"

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
        
        

# Create the boto3 client for translation
transL = boto3.client('translate', region_name='us-east-2', 
                        aws_access_key_id=AWS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET)


# Translate text
text = "This is the begining: Genesis"
translateMe = transL.translate_text(
    Text = text,
    SourceLanguageCode = 'auto',
    TargetLanguageCode = 'es')


# get the translated test
translateMe['TranslatedText']


# Create the boto3 client for comprehension
compreh = boto3.client('comprehend', region_name='us-east-2', 
                        aws_access_key_id=AWS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET)


# Detect the language
outcome = compreh.detect_dominant_language(
    Text = 'son de le homine d los')

# Analyse the sentiment of the word
senti = compreh.detect_sentiment(
    Text = text,
    LanguageCode = 'es')['Sentiment']




