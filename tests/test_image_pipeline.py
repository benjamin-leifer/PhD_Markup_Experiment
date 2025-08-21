from __future__ import annotations

# mypy: ignore-errors

import os
import tempfile
import importlib.util
import sys
import types
from typing import List
from io import BytesIO

from PIL import Image

TESTS_DIR = os.path.dirname(__file__)
PACKAGE_DIR = os.path.abspath(os.path.join(TESTS_DIR, "..", "Python_Codes", "BLeifer_Battery_Analysis", "battery_analysis"))

# Stub package structure to load image_pipeline without MongoDB
package_stub = types.ModuleType("battery_analysis")
package_stub.__path__ = [PACKAGE_DIR]  # type: ignore[attr-defined]
sys.modules["battery_analysis"] = package_stub


class RawDataFile:
    _id_counter = 0

    def __init__(self, file_data, filename="", file_type=None, sample=None, test_result=None, operator=None, tags=None, metadata=None):
        RawDataFile._id_counter += 1
        self.file_data = file_data
        self.filename = filename
        self.file_type = file_type
        self.sample = sample
        self.test_result = test_result
        self.operator = operator
        self.tags = tags
        self.metadata = metadata or {}
        self.id = f"id{RawDataFile._id_counter}"

    def save(self):
        pass


class Sample:
    def __init__(self):
        self.images: List[RawDataFile] = []
        self.saved = False

    def save(self):
        self.saved = True


class TestResult:  # pragma: no cover - placeholder
    pass


models_stub = types.ModuleType("battery_analysis.models")
models_stub.RawDataFile = RawDataFile
models_stub.Sample = Sample
models_stub.TestResult = TestResult
sys.modules["battery_analysis.models"] = models_stub

utils_stub = types.ModuleType("battery_analysis.utils")
utils_stub.__path__ = [os.path.join(PACKAGE_DIR, "utils")]
sys.modules["battery_analysis.utils"] = utils_stub

spec = importlib.util.spec_from_file_location(
    "battery_analysis.utils.image_pipeline",
    os.path.join(PACKAGE_DIR, "utils", "image_pipeline.py"),
)
image_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_pipeline)  # type: ignore

stored_files: List[RawDataFile] = []


def _objects(**query):
    results = []
    for rf in stored_files:
        match = True
        for key, value in query.items():
            if key == "metadata__source_file_id":
                if rf.metadata.get("source_file_id") != value:
                    match = False
                    break
            elif key == "tags":
                if not rf.tags or value not in rf.tags:
                    match = False
                    break
            elif key == "tags__all":
                if not rf.tags or any(tag not in rf.tags for tag in value):
                    match = False
                    break
            elif getattr(rf, key) != value:
                match = False
                break
        if match:
            results.append(rf)

    class _Query(list):
        def first(self):
            return self[0] if self else None

    return _Query(results)


RawDataFile.objects = staticmethod(_objects)  # type: ignore[attr-defined]


def fake_store_raw_data_file(path, **kwargs):
    with open(path, "rb") as f:
        data = f.read()
    raw = RawDataFile(BytesIO(data), filename=os.path.basename(path), **kwargs)
    stored_files.append(raw)
    return raw


def fake_get_raw_data_file_by_id(file_id, as_file_path=False):
    for rf in stored_files:
        if str(rf.id) == str(file_id):
            if as_file_path:
                tmp = tempfile.NamedTemporaryFile(delete=False)
                rf.file_data.seek(0)
                tmp.write(rf.file_data.read())
                tmp.close()
                return tmp.name
            rf.file_data.seek(0)
            return rf.file_data.read()
    raise KeyError(file_id)


image_pipeline.store_raw_data_file = fake_store_raw_data_file  # type: ignore
image_pipeline.get_raw_data_file_by_id = fake_get_raw_data_file_by_id  # type: ignore

ingest_image_file = image_pipeline.ingest_image_file
generate_thumbnail = image_pipeline.generate_thumbnail
search_images = image_pipeline.search_images


def _create_image(size=(100, 100)):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.new("RGB", size, color=(255, 0, 0)).save(tmp, format="PNG")
    tmp.close()
    return tmp.name


def test_generate_thumbnail_creation_and_metadata():
    stored_files.clear()
    img_path = _create_image((640, 480))
    try:
        raw = ingest_image_file(img_path)
        thumb = generate_thumbnail(raw)
        assert thumb.file_type == "thumbnail"
        assert thumb.tags == ["thumbnail"]
        assert thumb.metadata["width"] <= 256
        assert thumb.metadata["height"] <= 256
    finally:
        os.remove(img_path)


def test_ingest_image_file_generates_thumbnail():
    stored_files.clear()
    img_path = _create_image((300, 300))
    try:
        ingest_image_file(img_path, create_thumbnail=True)
        assert len(stored_files) == 2
        thumb = stored_files[1]
        assert thumb.file_type == "thumbnail"
        assert thumb.tags == ["thumbnail"]
    finally:
        os.remove(img_path)


def test_ingest_image_file_links_sample():
    stored_files.clear()
    img_path = _create_image((200, 200))
    sample = Sample()
    try:
        raw = ingest_image_file(img_path, sample=sample)
        assert raw in sample.images
        assert sample.saved
    finally:
        os.remove(img_path)


def test_search_images_filters_by_metadata():
    stored_files.clear()
    sample_a = Sample()
    sample_b = Sample()
    img1 = _create_image((10, 10))
    img2 = _create_image((10, 10))
    img3 = _create_image((10, 10))
    try:
        ingest_image_file(img1, sample=sample_a, tags=["foo"], operator="alice")
        ingest_image_file(img2, sample=sample_a, tags=["foo", "bar"], operator="bob")
        ingest_image_file(img3, sample=sample_b, tags=["bar"], operator="alice")

        assert set(search_images(sample=sample_a)) == {stored_files[0], stored_files[1]}
        assert set(search_images(tags="foo")) == {stored_files[0], stored_files[1]}
        assert set(search_images(tags=["foo", "bar"])) == {stored_files[1]}
        assert set(search_images(operator="alice")) == {stored_files[0], stored_files[2]}
        assert set(search_images(sample=sample_a, operator="alice")) == {stored_files[0]}
    finally:
        for p in (img1, img2, img3):
            os.remove(p)
