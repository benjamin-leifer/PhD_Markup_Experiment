from mongoengine import Document, fields


class RawDataFile(Document):  # type: ignore[misc]
    """Document to store the original raw data files using GridFS."""

    filename = fields.StringField(required=True)
    file_data = fields.FileField(required=True)
    file_type = fields.StringField()  # e.g., 'arbin_excel', 'biologic_mpt', etc.
    upload_date = fields.DateTimeField()
    test_result = fields.ReferenceField("TestResult")
    sample = fields.ReferenceField("Sample")
    operator = fields.StringField()
    acquisition_device = fields.StringField()
    tags = fields.ListField(fields.StringField())
    metadata = fields.DictField()

    meta = {
        "collection": "raw_data_files",
        "indexes": ["filename", "test_result", "sample", "tags", "operator"],
    }
