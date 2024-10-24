import pymongo
from pymongo import MongoClient
from tkinter import Tk, Label, Button, Entry


class DatabaseObject:
    def __init__(self, db_name, collection_name, host='localhost', port=27017):
        """Initialize the MongoDB connection and set the database and collection."""
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.document_id = None  # Optional: Track the document's ID

    # ---------------- MongoDB Methods ----------------

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

    def save_to_db(self):
        """Insert or update the object in the database."""
        document = self.to_dict()

        # If document_id exists, update the document; otherwise, insert a new one
        if self.document_id:
            query = {"_id": self.document_id}
            self.update_in_db(query, document)
        else:
            self.insert_document(document)

    def load_from_db(self, query):
        """Load a document from the collection based on a query and populate the object's attributes."""
        document = self.collection.find_one(query)
        if document:
            self.from_dict(document)
            self.log_action(f"Loaded document where {query}")
            return document
        return None

    def delete_from_db(self, query):
        """Delete a document from the collection based on a query."""
        result = self.collection.delete_one(query)
        self.log_action(f"Deleted document where {query}")
        return result

    def to_dict(self):
        """Convert the object to a dictionary. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement the 'to_dict' method")

    def from_dict(self, document):
        """Populate the object from a dictionary. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement the 'from_dict' method")

    # ---------------- Tkinter Methods ----------------

    def create_gui(self):
        """Create a simple Tkinter GUI to display the object's properties."""
        root = Tk()
        root.title(f"{self.__class__.__name__} Properties")

        # Add custom components from subclasses
        self.add_gui_components(root)

        button_save = Button(root, text="Save", command=lambda: self.save_from_gui(root))
        button_save.pack()

        button_close = Button(root, text="Close", command=root.quit)
        button_close.pack()

        root.mainloop()

    def add_gui_components(self, root):
        """Add custom Tkinter components for the subclass. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement 'add_gui_components' method")

    def save_from_gui(self, root):
        """Save data from the GUI to MongoDB."""
        # Example: Capture user input from the GUI (this can be customized per subclass)
        self.save_to_db()

    # ---------------- Helper Methods ----------------

    def log_action(self, action):
        """Log the actions for debugging purposes."""
        print(f"[LOG] {action}")

    def validate_fields(self):
        """Validate object fields before saving to MongoDB."""
        # Subclasses can implement specific validation checks
        pass

    def convert_units(self, from_unit, to_unit, value):
        """Convert units if necessary."""
        conversion_factors = {
            ('mAh', 'Ah'): 0.001,
            ('cm²', 'm²'): 0.0001
        }
        return value * conversion_factors.get((from_unit, to_unit), 1)

