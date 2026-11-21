"""Getting input files (images, LoRAs, checkpoints) from the user's computer.

Works in Google Colab (native file picker via ``google.colab.files.upload``) and falls back to an
``ipywidgets.FileUpload`` drag-and-drop widget in Colab or plain Jupyter. Outside a notebook, or when
neither is available, functions return ``None`` / an empty list and print a hint instead of failing.

Precedence used by :func:`resolve_image`: uploaded file > explicit source (URL, local/Drive path, ``last``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

UPLOAD_ROOT = Path(os.environ.get("AURORA_UPLOAD_DIR", "/content/uploads" if os.path.isdir("/content") else "uploads"))

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
WEIGHT_EXTS = (".safetensors", ".ckpt", ".pt", ".bin")

KIND_DIRS = {"image": "images", "lora": "loras", "checkpoint": "checkpoints"}
KIND_EXTS = {"image": IMAGE_EXTS, "lora": WEIGHT_EXTS, "checkpoint": WEIGHT_EXTS}
UPLOAD_TOKENS = ("upload", "uploaded", "@upload")


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _safe_name(name: str) -> str:
    name = os.path.basename(name or "upload")
    return re.sub(r"[^\w.\-]+", "_", name) or "upload"


def _unique(path: Path) -> Path:
    i = 1
    out = path
    while out.exists():
        out = path.with_name(f"{path.stem}-{i}{path.suffix}")
        i += 1
    return out


def save_bytes(name: str, data: bytes, kind: str = "image", root: Path | str | None = None) -> Path:
    """Write uploaded bytes to ``<root>/<kind dir>/<name>`` (unique name) and return the path."""
    dest = Path(root or UPLOAD_ROOT) / KIND_DIRS.get(kind, kind)
    dest.mkdir(parents=True, exist_ok=True)
    path = _unique(dest / _safe_name(name))
    path.write_bytes(bytes(data))
    return path


def check_extension(path: Path | str, kind: str) -> bool:
    exts = KIND_EXTS.get(kind)
    return exts is None or str(path).lower().endswith(exts)


def _filter(paths: list[Path], kind: str) -> list[Path]:
    good = [p for p in paths if check_extension(p, kind)]
    for p in paths:
        if p not in good:
            print(f"[aurora] Ignoring {p.name}: not a {kind} file ({', '.join(KIND_EXTS[kind])}).")
    return good


def colab_upload(kind: str = "image", root: Path | str | None = None) -> list[Path]:
    """Open Colab's native file picker (blocks until the user picks files). Returns saved paths."""
    if not in_colab():
        print("[aurora] Colab file picker not available here; use upload_widget() or pass a path/URL.")
        return []
    from google.colab import files

    dest = Path(root or UPLOAD_ROOT) / KIND_DIRS.get(kind, kind)
    dest.mkdir(parents=True, exist_ok=True)
    if kind != "image":
        print(
            "[aurora] Tip: files over ~2 GB upload slowly through the browser; put them on Google Drive and use the path instead."
        )
    try:
        uploaded = files.upload(target_dir=str(dest))
    except TypeError:  # older google.colab without target_dir
        uploaded = files.upload()
    paths: list[Path] = []
    for name, data in uploaded.items():
        p = Path(name)
        if p.exists() and p.parent.resolve() == dest.resolve():
            paths.append(p)
        else:  # older Colab saved into the cwd (or not at all): keep our own copy
            paths.append(save_bytes(p.name, data, kind, root))
            if p.exists() and p.parent.resolve() != dest.resolve():
                try:
                    p.unlink()
                except OSError:
                    pass
    return _filter(paths, kind)


def _widget_items(value) -> list[tuple[str, bytes]]:
    """Normalise ipywidgets FileUpload.value across ipywidgets 7 (dict) and 8 (tuple of dicts)."""
    items: list[tuple[str, bytes]] = []
    if not value:
        return items
    if isinstance(value, dict):  # ipywidgets 7: {name: {"metadata": ..., "content": bytes}}
        for name, info in value.items():
            items.append((info.get("metadata", {}).get("name", name), bytes(info["content"])))
    else:  # ipywidgets 8: ({"name": ..., "content": memoryview, ...}, ...)
        for info in value:
            info = dict(info) if not isinstance(info, dict) else info
            items.append((info["name"], bytes(info["content"])))
    return items


class Uploader:
    """Collects files uploaded from the user's computer for one purpose (``kind``).

    ``pick()`` opens Colab's file picker; ``widget()`` shows a drag-and-drop FileUpload widget whose
    files are read lazily by ``latest()`` / ``paths()``, so the user can drop a file and simply run
    the next cell.
    """

    def __init__(self, kind: str = "image", root: Path | str | None = None):
        if kind not in KIND_DIRS:
            raise ValueError(f"kind must be one of {tuple(KIND_DIRS)}")
        self.kind = kind
        self.root = root
        self._paths: list[Path] = []
        self._widget = None
        self._widget_seen: set[tuple[str, int]] = set()

    # -- sources ----------------------------------------------------------------------------
    def pick(self) -> list[Path]:
        new = colab_upload(self.kind, self.root)
        self._paths.extend(new)
        for p in new:
            print(f"[aurora] Uploaded {self.kind}: {p}")
        return new

    def widget(self):
        """Display a drag-and-drop upload widget. Returns the widget, or None if ipywidgets is missing."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print("[aurora] ipywidgets not installed; use the Colab file picker or a path/URL instead.")
            return None
        if in_colab():
            try:
                from google.colab import output

                output.enable_custom_widget_manager()
            except Exception:
                pass
        accept = ",".join(KIND_EXTS[self.kind])
        self._widget = widgets.FileUpload(accept=accept, multiple=True, description=f"Drop {self.kind}(s)")
        display(self._widget)
        print("Drop or select file(s) above, then run the next cell.")
        return self._widget

    def add(self, path: Path | str) -> Path:
        """Register an already-saved file (e.g. from tests or another source)."""
        p = Path(path)
        self._paths.append(p)
        return p

    def _sync_widget(self) -> None:
        if self._widget is None:
            return
        for name, data in _widget_items(self._widget.value):
            key = (name, len(data))
            if key in self._widget_seen:
                continue
            self._widget_seen.add(key)
            p = save_bytes(name, data, self.kind, self.root)
            if check_extension(p, self.kind):
                self._paths.append(p)
                print(f"[aurora] Uploaded {self.kind}: {p}")
            else:
                print(f"[aurora] Ignoring {name}: not a {self.kind} file.")

    # -- access -----------------------------------------------------------------------------
    def paths(self) -> list[Path]:
        self._sync_widget()
        return [p for p in self._paths if p.exists()]

    def latest(self) -> Path | None:
        paths = self.paths()
        return paths[-1] if paths else None

    def clear(self) -> None:
        self._paths.clear()
        self._widget_seen.clear()
        if self._widget is not None:
            try:
                self._widget.value = () if not isinstance(self._widget.value, dict) else {}
            except Exception:
                pass

    def __bool__(self) -> bool:
        return self.latest() is not None

    def __repr__(self) -> str:
        return f"Uploader(kind={self.kind!r}, files={[p.name for p in self.paths()]})"


def upload(kind: str = "image", method: str = "file picker", uploader: Uploader | None = None) -> Uploader:
    """Notebook helper: ``method`` is "file picker" (Colab dialog) or "drag & drop widget"."""
    up = uploader or Uploader(kind)
    if method.startswith("drag") or not in_colab():
        if up.widget() is None and in_colab():
            up.pick()
    else:
        up.pick()
    return up


def _parse_token(src: str | None) -> int | None:
    """``"uploaded"`` -> -1 (latest); ``"uploaded:2"`` -> 1 (0-based index); anything else -> None."""
    text = (src or "").strip().lower()
    if text in UPLOAD_TOKENS:
        return -1
    m = re.fullmatch(r"(?:@?upload(?:ed)?)\s*[:#]\s*(\d+)", text)
    if m and int(m.group(1)) >= 1:
        return int(m.group(1)) - 1
    return None


def is_upload_token(src: str | None) -> bool:
    return _parse_token(src) is not None


def resolve_source(src: str | None, uploader: Uploader | None = None, kind: str = "lora") -> str | None:
    """For LoRA/checkpoint fields.

    ``"uploaded"`` -> latest uploaded file, ``"uploaded:N"`` -> N-th uploaded file (1-based);
    anything else is returned stripped (``None`` if blank).
    """
    idx = _parse_token(src)
    if idx is None:
        return (src or "").strip() or None
    paths = uploader.paths() if uploader is not None else []
    if not paths:
        raise FileNotFoundError(f"'{src}' given but no {kind} file was uploaded; run the upload cell first.")
    if idx >= len(paths):
        raise IndexError(f"'{src}' given but only {len(paths)} {kind} file(s) were uploaded: {[p.name for p in paths]}")
    return str(paths[idx])


def resolve_image(source=None, uploader: Uploader | None = None, last=None, allow_picker: bool = True):
    """Return a PIL image using the precedence: uploaded file > ``source`` > Colab file picker.

    ``source`` may be a PIL image, URL, local path (``/content/...``), Drive path
    (``/content/drive/MyDrive/...``), or ``"last"`` (uses ``last``). Returns ``None`` if nothing is available.
    """
    from PIL import Image, ImageOps

    def _open(p):
        img = Image.open(p)
        return ImageOps.exif_transpose(img).convert("RGB")

    if uploader is not None:
        latest = uploader.latest()
        if latest is not None:
            print(f"[aurora] Using uploaded image {latest.name}")
            return _open(latest)
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    src = (source or "").strip() if isinstance(source, str) else source
    if src:
        if isinstance(src, str) and src.lower() == "last":
            if last is None:
                raise ValueError("'last' requested but there is no previous image yet.")
            return last.convert("RGB")
        if isinstance(src, str) and re.match(r"^https?://", src):
            from diffusers.utils import load_image

            return load_image(src).convert("RGB")
        path = Path(src).expanduser()
        if path.is_file():
            return _open(path)
        if src.startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
            from .outputs import mount_drive

            mount_drive()
            if path.is_file():
                return _open(path)
        raise FileNotFoundError(f"Image not found: {src}")
    if allow_picker and in_colab():
        print("[aurora] No image given: pick one from your computer.")
        up = uploader or Uploader("image")
        new = up.pick()
        if new:
            return _open(new[-1])
    return None


__all__ = [
    "IMAGE_EXTS",
    "UPLOAD_ROOT",
    "WEIGHT_EXTS",
    "Uploader",
    "colab_upload",
    "in_colab",
    "is_upload_token",
    "resolve_image",
    "resolve_source",
    "save_bytes",
    "upload",
]
