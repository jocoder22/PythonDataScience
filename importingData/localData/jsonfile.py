#!/usr/bin/env python
import json

with open('myfile.json', 'r') as json_file:
    jsonData = json.load(json_file)

type(jsonData) ## dict

for key, value in jsonData.items():
    print(f'{key} : {value}')


# Load JSON: json_data
with open("a_movie.json") as json_file:
    json_data = json.load(json_file)

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])



