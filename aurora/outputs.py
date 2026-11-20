"""Saving outputs locally and (optionally) to Google Drive."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("AURORA_OUTPUT_DIR", "/content/aurora_outputs" if os.path.isdir("/content") else "aurora_outputs")
)
DRIVE_ROOT = Path("/content/drive")


def slugify(text: str, max_len: int = 48) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").lower()).strip("-")
    return (text[:max_len].rstrip("-")) or "untitled"


def make_output_path(prompt: str, ext: str, kind: str, seed: int | None = None, out_dir: Path | str | None = None) -> Path:
    out = Path(out_dir or DEFAULT_OUTPUT_DIR) / kind
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    seed_part = f"-s{seed}" if seed is not None else ""
    path = out / f"{stamp}-{slugify(prompt)}{seed_part}.{ext.lstrip('.')}"
    i = 1
    while path.exists():
        path = path.with_name(f"{path.stem}-{i}{path.suffix}")
        i += 1
    return path


def write_metadata(path: Path | str, metadata: dict) -> Path:
    meta_path = Path(path).with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    return meta_path


def save_image(image, prompt: str, seed: int | None = None, metadata: dict | None = None, out_dir=None) -> Path:
    """Save a PIL image as PNG with generation parameters embedded in the PNG text chunk."""
    from PIL.PngImagePlugin import PngInfo

    path = make_output_path(prompt, "png", "images", seed, out_dir)
    info = PngInfo()
    if metadata:
        info.add_text("aurora", json.dumps(metadata, default=str))
        info.add_text("parameters", str(metadata.get("prompt", prompt)))
    image.save(path, pnginfo=info)
    if metadata:
        write_metadata(path, metadata)
    return path


def save_video(frames, prompt: str, fps: int = 16, seed: int | None = None, metadata: dict | None = None, out_dir=None) -> Path:
    """Export frames (list of PIL images or ndarray) to MP4 via diffusers' export_to_video."""
    from diffusers.utils import export_to_video

    path = make_output_path(prompt, "mp4", "videos", seed, out_dir)
    export_to_video(frames, str(path), fps=fps)
    if metadata:
        write_metadata(path, metadata)
    return path


def mount_drive() -> bool:
    try:
        from google.colab import drive
    except ImportError:
        print("[aurora] Not running in Colab; Drive not available.")
        return False
    if not (DRIVE_ROOT / "MyDrive").exists():
        drive.mount(str(DRIVE_ROOT))
    return (DRIVE_ROOT / "MyDrive").exists()


def copy_to_drive(path: Path | str, subdir: str = "Aurora") -> Path | None:
    """Copy a file (and its .json sidecar) to MyDrive/<subdir>/. Mounts Drive if needed."""
    if not mount_drive():
        return None
    dest_dir = DRIVE_ROOT / "MyDrive" / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = Path(path)
    dest = dest_dir / path.name
    shutil.copy2(path, dest)
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        shutil.copy2(sidecar, dest_dir / sidecar.name)
    print(f"[aurora] Saved to Drive: {dest}")
    return dest


def show_video(path: Path | str, width: int = 640):
    """Display an MP4 inline in a notebook."""
    from base64 import b64encode

    from IPython.display import HTML, display

    data = b64encode(Path(path).read_bytes()).decode()
    display(
        HTML(
            f'<video width="{width}" controls autoplay loop muted><source src="data:video/mp4;base64,{data}" type="video/mp4"></video>'
        )
    )


def show_images(images, cols: int = 2, width: int = 384):
    import io
    from base64 import b64encode

    from IPython.display import HTML, display

    cells = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cells.append(
            f'<img src="data:image/png;base64,{b64encode(buf.getvalue()).decode()}" style="width:{width}px;margin:4px"/>'
        )
    display(HTML("<div style='display:flex;flex-wrap:wrap'>" + "".join(cells) + "</div>"))
