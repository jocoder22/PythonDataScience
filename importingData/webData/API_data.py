#!/usr/bin/env python
# Import requests package
import requests
import json

# Assign URL to variable: url
url = 'http://www.omdbapi.com/?apikey=72bc447a&t=the+social+network'

# Package the request, send the request and catch the response: r
r = requests.get(url)
result_text = r.text
jdata = r.json()
# Print the text of the response
print(result_text)

print(type(result_text)) # <class 'str'>

n = 1
for key in jdata.keys() :
    print(f'{n:>3}. {key:11}: {jdata[key]} ')
    n += 1




# Assign URL to variable: url

url = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza'
# Package the request, send the request and catch the response: r
r = requests.get(url)

# Decode the JSON data into a dictionary: json_data

json_data = r.json()
# Print the Wikipedia page extract
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)
