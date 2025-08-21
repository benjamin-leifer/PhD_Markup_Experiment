# Image Ingestion Pipeline

This project provides helpers for storing microscope images and quickly retrieving
them later. The workflow is:

1. **Ingest** the raw image using `ingest_image_file`.
2. **Generate** a small thumbnail for quick previews.
3. **Search** and retrieve the stored images or their thumbnails.

## Ingestion

`ingest_image_file` validates the image, stores it in GridFS, and optionally
links it to a `Sample`::

```python
from battery_analysis.models import Sample
from battery_analysis.utils import ingest_image_file

sample = Sample()
raw = ingest_image_file("microscope.png", sample=sample, create_thumbnail=True)
```

The returned `RawDataFile` is appended to `sample.images` and can later be found
with `search_images`.

## Thumbnails

A 256×256 thumbnail is created automatically when `create_thumbnail=True` during
ingestion. Thumbnails may also be generated manually:

```python
from battery_analysis.utils.image_pipeline import generate_thumbnail

thumb = generate_thumbnail(raw)
```

Both helpers record the thumbnail's dimensions and link it to the source image
through `metadata['source_file_id']`.

## Retrieval and Search

Stored images and their thumbnails can be accessed by id:

```python
from battery_analysis.utils.image_pipeline import get_image, get_thumbnail, search_images

image_bytes = get_image(raw.id)
thumb_bytes = get_thumbnail(raw)
results = search_images(sample=sample)
```

`search_images` filters by `sample`, `tags`, or `operator`, returning a
queryset of matching image records.
