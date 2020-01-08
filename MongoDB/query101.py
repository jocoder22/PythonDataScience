import requests
from pymongo import MongoClient

sp = {"sep":"\n\n", "end":"\n\n"}

client = MongoClient()

db = client["nobelprizes"]

# Save a list of names of the collections managed by the "nobelprizes" database
nobel_coll_names = client.nobelprizes.list_collection_names()
print(nobel_coll_names, **sp)


# Get the fields present in each type of document
prize_fields = ____(____.keys())
laureate_fields = ____(____.keys())

print(prize_fields)
print(laureate_fields)