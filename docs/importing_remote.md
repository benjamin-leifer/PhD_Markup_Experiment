# Importing Data from Remote Storage

The `battery_analysis.utils.remote_import` module can retrieve data files from
remote locations before passing them to the standard `import_directory` flow.
It supports SFTP and Amazon S3 locations.

## Credentials

### SFTP

Pass the remote location as `sftp://user@host/path`.  The helper connects using
[`paramiko`](https://www.paramiko.org/).  Passwords may be supplied via the
`SFTP_PASSWORD` environment variable or standard SSH key configuration.

### S3

For S3 locations provide a URI like `s3://bucket/prefix`.  Authentication uses
[`boto3`](https://boto3.amazonaws.com) which follows the usual AWS credential
resolution (environment variables such as `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`, configuration files, or IAM roles).

## CLI Usage

The `import_directory` utility accepts a `--remote` option which downloads the
remote tree to a temporary directory before processing.

```bash
python -m battery_analysis.utils.import_directory --remote s3://my-bucket/data --dry-run
python -m battery_analysis.utils.import_directory --remote sftp://user@example.com/path
```

The temporary files are removed automatically once processing completes.
