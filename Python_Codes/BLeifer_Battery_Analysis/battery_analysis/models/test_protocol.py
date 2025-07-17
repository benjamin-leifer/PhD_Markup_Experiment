from mongoengine import Document, fields

class TestProtocol(Document):
    """Simple document to store test protocol definitions."""

    name = fields.StringField(required=True, unique=True)
    summary = fields.StringField(required=True)
    c_rates = fields.ListField(fields.FloatField(), default=list)

    meta = {"collection": "test_protocols"}

    def __str__(self):
        return f"<TestProtocol {self.name}>"
