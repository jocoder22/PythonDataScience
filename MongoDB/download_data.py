import requests
from pymongo import MongoClient

sp = {"sep":"\n\n", "end":"\n\n"}

# https://www.tutorialspoint.com/mongodb/mongodb_drop_database.htm
# first on commmand prompt type : mongod, then open another command prompt type: mongo
client = MongoClient()

db = client["nobelprizes"]
print(db.name, **sp)

for collections_m in ["prizes", "laureates", "countries"]:
    db[collections_m].delete_many({})


for collections_m in ["prizes" , "countrys" , "laureates"]:
    response = requests.get(
        "http://api.nobelprize.org/v1/{}.json".format(collections_m[:-1])
    )
    
    if collections_m == "countrys":
        documents = response.json()["countries"]
        db["countries"].insert_many(documents)
        continue
        
    
    documents = response.json()[collections_m]
    db[collections_m].insert_many(documents)


# Save a list of names of the databases managed by client
db_names = client.list_database_names()
print(db_names, **sp)

# Save a list of names of the collections managed by the "nobel" database
nobel_coll_names = client.nobelprizes.list_collection_names()
print(nobel_coll_names, **sp)

filter = {}

for collections_m in ["prizes", "laureates", "countries"]:
    bb = db[collections_m].count_documents({})
    print(f'The {collections_m} has {bb} rows', **sp)

doc = db.prizes.find_one(filter)
print(doc, **sp)


# Retrieve sample prize and laureate documents
prize_one = db.prizes.find_one()
laureate_one = db.laureates.find_one()

# Print the sample prize and laureate documents
print(prize_one, **sp)
print(laureate_one, **sp)
print(type(laureate_one), type(laureate_one), **sp)


