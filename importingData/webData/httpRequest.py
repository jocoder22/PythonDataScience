#!/usr/bin/env python

# Import packages
from urllib.request import urlopen, Request

# Specify the url
url = "http://www.datacamp.com/teach/documentation"

# This packages the request: request
request = Request(url)

# Sends the request and catches the response: response
response = urlopen(request)

# Print the datatype of response
print(type(response))
# <class 'http.client.HTTPResponse'>

# Extract the response: html
html = response.read()

# Print the html
print(html)

# Be polite and close the response!
response.close()




# Performing HTTP requests in Python using requests
# Note that unlike in the previous exercises using urllib, 
# you don't have to close the connection when using requests!
# Import package
import requests

# Specify the url: url
url = "http://www.datacamp.com/teach/documentation"

# Packages the request, send the request and catch the response: r
r = requests.get(url)

# Extract the response: text
text = r.text

