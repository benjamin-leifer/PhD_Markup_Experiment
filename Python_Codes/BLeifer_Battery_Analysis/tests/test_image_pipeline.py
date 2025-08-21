from __future__ import annotations

# mypy: ignore-errors

import io
import os
import tempfile
import importlib.util
import types
import sys

from PIL import Image
import numpy as np

TESTS_DIR = os.path.dirname(__file__)
PACKAGE_DIR = os.path.abspath(os.path.join(TESTS_DIR, "..", "battery_analysis"))

# Create stub package and models to satisfy imports without MongoDB dependencies
package_stub = types.ModuleType("battery_analysis")
package_stub.__path__ = [PACKAGE_DIR]
sys.modules["battery_analysis"] = package_stub


class RawDataFile:
    _id_counter = 0

    def __init__(
        self,
        file_data,
        filename="",
        file_type=None,
        sample=None,
        test_result=None,
        operator=None,
        tags=None,
        metadata=None,
    ):
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
        def first(self):  # pragma: no cover - trivial
            return self[0] if self else None

    return _Query(results)


RawDataFile.objects = staticmethod(_objects)  # type: ignore[attr-defined]


def fake_store_raw_data_file(path, **kwargs):
    with open(path, "rb") as f:
        data = f.read()
    raw = RawDataFile(
        file_data=io.BytesIO(data),
        filename=os.path.basename(path),
        file_type=kwargs.get("file_type"),
        sample=kwargs.get("sample"),
        test_result=kwargs.get("test_result"),
        operator=kwargs.get("operator"),
        tags=kwargs.get("tags"),
        metadata=kwargs.get("metadata", {}),
    )
    stored_files.append(raw)
    return raw


image_pipeline.store_raw_data_file = fake_store_raw_data_file


def fake_get_raw_data_file_by_id(file_id, as_file_path=False):
    rf = next((r for r in stored_files if r.id == file_id), None)
    if rf is None:
        raise ValueError("not found")
    data = rf.file_data.getvalue()
    if as_file_path:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(data)
        tmp.close()
        return tmp.name
    return data


image_pipeline.get_raw_data_file_by_id = fake_get_raw_data_file_by_id
generate_thumbnail = image_pipeline.generate_thumbnail
ingest_image_file = image_pipeline.ingest_image_file
get_image = image_pipeline.get_image
get_thumbnail = image_pipeline.get_thumbnail
process_image = image_pipeline.process_image
search_images = image_pipeline.search_images


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


def test_get_image_and_thumbnail():
    stored_files.clear()
    img_path = _create_image((128, 128))
    path = ""
    try:
        raw = ingest_image_file(img_path, create_thumbnail=True)

        data = get_image(raw.id)
        assert isinstance(data, bytes)

        path = get_image(raw, as_file_path=True)
        assert os.path.exists(path)

        thumb_data = get_thumbnail(raw)
        assert isinstance(thumb_data, bytes)
    finally:
        os.remove(img_path)
        if path and os.path.exists(path):
            os.remove(path)


def test_process_image_invokes_preprocess():
    stored_files.clear()
    img_path = _create_image((20, 20))
    try:
        raw = ingest_image_file(img_path)

        called = {"count": 0}

        def preprocess(arr):
            called["count"] += 1
            return arr[::2, ::2]

        result = process_image(raw, preprocess=preprocess)
        assert called["count"] == 1
        assert isinstance(result, np.ndarray)
        assert result.shape[:2] == (10, 10)
    finally:
        os.remove(img_path)


def test_process_image_preserves_original_bytes():
    stored_files.clear()
    img_path = _create_image((20, 20))
    try:
        raw = ingest_image_file(img_path)
        before = get_image(raw)

        def preprocess(arr):
            arr[:] = 0
            return arr

        process_image(raw, preprocess=preprocess)
        after = get_image(raw)
        assert before == after
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
