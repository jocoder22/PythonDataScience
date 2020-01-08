import requests
from pymongo import MongoClient

sp = {"sep":"\n\n", "end":"\n\n"}

client = MongoClient()

db = client["nobelprizes"]

# Save a list of names of the collections managed by the "nobelprizes" database
nobel_coll_names = client.nobelprizes.list_collection_names()
print(nobel_coll_names, **sp)


# Retrieve sample prize, countries, and laureate documents
prize_doc = db.prizes.find_one()
country_doc = db.country.find_one()
laureate_doc = db.laureates.find_one()

# Print the sample prize, countries, and laureate documents
print(prize_doc, country_doc, laureate_doc, **sp)
print(type(prize_doc), type(country_doc), type(laureate_doc), **sp)
