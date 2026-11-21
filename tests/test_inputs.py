"""Upload helpers outside Colab: graceful fallbacks, precedence and token resolution."""

import pytest
from PIL import Image

from aurora import inputs
from aurora.inputs import Uploader, _widget_items, is_upload_token, resolve_image, resolve_source, save_bytes


def _png(path, color=(255, 0, 0), size=(32, 16)):
    Image.new("RGB", size, color).save(path)
    return path


def test_not_in_colab_picker_is_graceful(tmp_path, capsys):
    assert not inputs.in_colab()
    assert inputs.colab_upload("image", tmp_path) == []
    assert "not available" in capsys.readouterr().out


def test_save_bytes_unique_and_sanitised(tmp_path):
    a = save_bytes("my photo (1).png", b"x", "image", tmp_path)
    b = save_bytes("my photo (1).png", b"y", "image", tmp_path)
    assert a.parent == tmp_path / "images"
    assert a != b and a.exists() and b.exists()
    assert " " not in a.name and "(" not in a.name


def test_widget_value_formats():
    v7 = {"a.png": {"metadata": {"name": "a.png"}, "content": b"123"}}
    v8 = ({"name": "b.png", "content": memoryview(b"45"), "size": 2},)
    assert _widget_items(v7) == [("a.png", b"123")]
    assert _widget_items(v8) == [("b.png", b"45")]
    assert _widget_items(()) == []


class FakeWidget:
    def __init__(self, value):
        self.value = value


def test_uploader_reads_widget_lazily(tmp_path):
    up = Uploader("image", root=tmp_path)
    assert up.latest() is None and not up
    up._widget = FakeWidget(({"name": "drop.png", "content": b"img"}, {"name": "notes.txt", "content": b"t"}))
    latest = up.latest()
    assert latest is not None and latest.name == "drop.png"
    assert len(up.paths()) == 1  # .txt ignored, no duplicates on re-read
    up.clear()
    assert up.latest() is None


def test_uploaded_image_takes_precedence(tmp_path):
    local = _png(tmp_path / "local.png", (0, 255, 0))
    up = Uploader("image", root=tmp_path)
    assert resolve_image(str(local), uploader=up).getpixel((0, 0)) == (0, 255, 0)  # nothing uploaded yet
    up.add(_png(tmp_path / "uploaded.png", (0, 0, 255)))
    assert resolve_image(str(local), uploader=up).getpixel((0, 0)) == (0, 0, 255)
    assert resolve_image(str(local), uploader=None).getpixel((0, 0)) == (0, 255, 0)


def test_resolve_image_fallbacks(tmp_path):
    last = Image.new("RGB", (8, 8), (9, 9, 9))
    assert resolve_image("last", last=last).getpixel((0, 0)) == (9, 9, 9)
    assert resolve_image("", allow_picker=True) is None  # outside Colab: no picker, no crash
    assert resolve_image(last).size == (8, 8)
    with pytest.raises(FileNotFoundError):
        resolve_image(str(tmp_path / "missing.png"))
    with pytest.raises(ValueError):
        resolve_image("last", last=None)


def test_upload_tokens(tmp_path):
    assert is_upload_token("uploaded") and is_upload_token(" Upload ") and is_upload_token("uploaded:2")
    assert not is_upload_token("owner/repo") and not is_upload_token("") and not is_upload_token("uploaded:0")
    up = Uploader("lora", root=tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_source("uploaded", up)
    a = up.add(save_bytes("a.safetensors", b"1", "lora", tmp_path))
    b = up.add(save_bytes("b.safetensors", b"2", "lora", tmp_path))
    assert resolve_source("uploaded", up) == str(b)
    assert resolve_source("uploaded:1", up) == str(a)
    with pytest.raises(IndexError):
        resolve_source("uploaded:3", up)
    assert resolve_source("  owner/repo ", up) == "owner/repo"
    assert resolve_source("", up) is None


def test_uploader_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Uploader("video")
