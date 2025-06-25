#!/usr/bin/env python3
import os, json
from pymongo import MongoClient

# 1) Connect (replace with your URI or host/port/db)
client = MongoClient("mongodb://USER:PASS@localhost:27017/")
db = client["your_db"]

# 2) Prepare output folder
out_dir = "data"
os.makedirs(out_dir, exist_ok=True)

# 3) Loop through collections
for coll_name in db.list_collection_names():
    coll = db[coll_name]
    # OPTION: sample N docs instead of full collection
    cursor = coll.find().limit(10)
    docs = []
    for doc in cursor:
        doc.pop("_id", None)            # drop Mongo’s ObjectId if you want
        # TODO: redact fields here if needed
        docs.append(doc)
    # 4) Write JSON file
    with open(os.path.join(out_dir, f"{coll_name}.json"), "w") as f:
        json.dump(docs, f, indent=2)
    print(f"Exported {len(docs)} docs from {coll_name}")
