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

for collections_m in ["prizes", "laureates", "countries"]:
    bb = db[collections_m].count_documents({})
    print(f'The {collections_m} has {bb} rows', end="\n\n")

doc = db.prizes.find_one(filter)
print(doc)


