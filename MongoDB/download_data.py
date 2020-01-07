import requests
from pymongo import MongoClient

# https://www.tutorialspoint.com/mongodb/mongodb_drop_database.htm
# first on commmand prompt type : mongod, then open another command prompt type: mongo
client = MongoClient()

db = client["nobelprizes"]
print(db.name)

# for collections_m in ["prizes", "laureates"]:
#     db[collections_m].delete_many({})


# for collections_m in ["prizes", "laureates", "countrys"]:
#     response = requests.get(
#         "http://api.nobelprize.org/v1/{}.json".format(collections_m[:-1])
#     )
    
#     if collections_m == "countrys":
#         documents = response.json()["countries"]
#         db["countries"].insert_many(documents)
        
#     else:
#         documents = response.json()[collections_m]
#         db[collections_m].insert_many(documents)


filter = {}
print(db.prizes.count_documents({}))

doc = db.prizes.find_one(filter)
print(doc)


# Save a list of names of the databases managed by client
db_names = client.list_database_names()
print(db_names)

# Save a list of names of the collections managed by the "nobel" database
nobel_coll_names = client.nobelprizes.list_collection_names()
print(nobel_coll_names)