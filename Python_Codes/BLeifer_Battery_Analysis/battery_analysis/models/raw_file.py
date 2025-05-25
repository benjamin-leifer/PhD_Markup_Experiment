from mongoengine import Document, fields


class RawDataFile(Document):
    """Document to store the original raw data files using GridFS."""
    filename = fields.StringField(required=True)
    file_data = fields.FileField(required=True)
    file_type = fields.StringField()  # e.g., 'arbin_excel', 'biologic_mpt', etc.
    upload_date = fields.DateTimeField()
    test_result = fields.ReferenceField('TestResult')

    meta = {
        'collection': 'raw_data_files',
        'indexes': [
            'filename',
            'test_result'
        ]
    }