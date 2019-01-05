#!/usr/bin/env python
import json

with open('myfile.json', 'r') as json_file:
    jsonData = json.load(json_file)

type(jsonData) ## dict

for key, value in jsonData.items():
    print(f'{key} : {value}')


