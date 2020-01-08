import requests
from pymongo import MongoClient

client = MongoClient()

db = client["nobelprizes"]

# Save a list of names of the collections managed by the "nobelprizes" database
nobel_coll_names = client.nobelprizes.list_collection_names()
print(nobel_coll_names)