from pymongo import MongoClient

class DatabaseObject:
    def __init__(self, db_name, collection_name, host='localhost', port=27017):
        """Initialize the MongoDB connection and set the database and collection."""
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.document_id = None  # Track the document's ID

    def insert_document(self, document):
        """Insert a new document into the collection."""
        result = self.collection.insert_one(document)
        self.document_id = result.inserted_id
        self.log_action(f"Inserted document with ID: {self.document_id}")
        return self.document_id

    def update_in_db(self, query, new_values):
        """Update an existing document in the collection based on a query."""
        result = self.collection.update_one(query, {'$set': new_values})
        self.log_action(f"Updated document where {query}")
        return result

    def find_duplicate(self, document):
        """Check if an identical document already exists in the collection."""
        duplicate = self.collection.find_one(document)
        return duplicate

    def save_to_db(self):
        """Insert a new document only if it is different from existing entries."""
        document = self.to_dict()  # Convert the object to a dictionary for MongoDB

        # Check for duplicates
        duplicate = self.find_duplicate(document)
        if duplicate:
            self.document_id = duplicate["_id"]
            self.log_action(f"Duplicate document found with ID: {self.document_id}. Skipping insert.")
            return self.document_id
        else:
            # If no duplicate is found, insert the document
            self.document_id = self.insert_document(document)
            self.log_action(f"Inserted new document with ID: {self.document_id}")
            return self.document_id

    @classmethod
    def load_from_db(cls, query):
        """Load a document from the collection based on a query and return an instance of the class."""
        client = MongoClient()  # Using default settings; update if necessary
        db = client[cls.db_name]  # Access class-level db_name
        collection = db[cls.collection_name]  # Access class-level collection_name

        document = collection.find_one(query)
        if document:
            instance = cls.__new__(cls)  # Create a new instance without calling __init__
            instance.from_dict(document)  # Populate instance with data
            instance.document_id = document.get("_id")
            instance.client = client  # Attach MongoDB client to instance
            instance.db = db  # Attach db reference to instance
            instance.collection = collection  # Attach collection reference to instance
            return instance
        return None

    def to_dict(self):
        """Convert the object to a dictionary. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement the 'to_dict' method")

    def from_dict(self, document):
        """Populate the object from a dictionary. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement the 'from_dict' method")

    def log_action(self, action, document=None):
        """Log the actions for debugging purposes. If a document is provided, print a summary."""
        print(f"[LOG] {action}")
        if document:
            print("[LOG] Document Summary:")
            for key, value in document.items():
                print(f"  {key}: {value}")