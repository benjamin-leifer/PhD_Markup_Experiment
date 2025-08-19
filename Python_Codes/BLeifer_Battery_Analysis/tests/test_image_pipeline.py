from __future__ import annotations

# mypy: ignore-errors

import io
import os
import tempfile
import importlib.util
import types
import sys

from PIL import Image

TESTS_DIR = os.path.dirname(__file__)
PACKAGE_DIR = os.path.abspath(os.path.join(TESTS_DIR, "..", "battery_analysis"))

# Create stub package and models to satisfy imports without MongoDB dependencies
package_stub = types.ModuleType("battery_analysis")
package_stub.__path__ = [PACKAGE_DIR]
sys.modules["battery_analysis"] = package_stub


class RawDataFile:
    def __init__(
        self,
        file_data,
        filename="",
        file_type=None,
        sample=None,
        test_result=None,
        tags=None,
        metadata=None,
    ):
        self.file_data = file_data
        self.filename = filename
        self.file_type = file_type
        self.sample = sample
        self.test_result = test_result
        self.tags = tags
        self.metadata = metadata or {}
        self.id = "dummy"

    def save(self):
        pass


class TestResult:  # pragma: no cover - placeholder
    pass


class Sample:
    def __init__(self):
        self.images: list = []
        self.saved = False

    def save(self):
        self.saved = True


models_stub = types.ModuleType("battery_analysis.models")
models_stub.RawDataFile = RawDataFile
models_stub.TestResult = TestResult
models_stub.Sample = Sample
sys.modules["battery_analysis.models"] = models_stub

utils_stub = types.ModuleType("battery_analysis.utils")
utils_stub.__path__ = [os.path.join(PACKAGE_DIR, "utils")]
sys.modules["battery_analysis.utils"] = utils_stub

spec = importlib.util.spec_from_file_location(
    "battery_analysis.utils.image_pipeline",
    os.path.join(PACKAGE_DIR, "utils", "image_pipeline.py"),
)
image_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_pipeline)

stored_files: list[RawDataFile] = []


def fake_store_raw_data_file(path, **kwargs):
    with open(path, "rb") as f:
        data = f.read()
    raw = RawDataFile(
        file_data=io.BytesIO(data),
        filename=os.path.basename(path),
        file_type=kwargs.get("file_type"),
        sample=kwargs.get("sample"),
        test_result=kwargs.get("test_result"),
        tags=kwargs.get("tags"),
        metadata={},
    )
    stored_files.append(raw)
    return raw


image_pipeline.store_raw_data_file = fake_store_raw_data_file

generate_thumbnail = image_pipeline.generate_thumbnail
ingest_image_file = image_pipeline.ingest_image_file


def _create_image(size=(512, 512)):
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
