"""Download checkpoints / LoRAs from Hugging Face, Civitai or plain URLs."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

MODELS_DIR = Path(os.environ.get("AURORA_MODELS_DIR", "/content/models" if os.path.isdir("/content") else "models"))

_HF_BLOB = re.compile(r"https?://huggingface\.co/([^/]+/[^/]+)/(?:blob|resolve)/([^/]+)/(.+)")


def _colab_secret(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    try:
        from google.colab import userdata

        return userdata.get(name)
    except Exception:
        return None


def is_url(src: str) -> bool:
    return bool(re.match(r"^https?://", src or ""))


def resolve_local(src: str, subdir: str = "checkpoints") -> str:
    """Return a local file path (or HF repo id) for ``src``.

    * HF repo ids (``owner/name``) and local paths are returned unchanged.
    * ``huggingface.co/.../blob|resolve/...`` URLs are fetched with ``hf_hub_download`` (uses HF_TOKEN).
    * Civitai URLs (``civitai.com/api/download/models/<id>`` or a model page with ``?modelVersionId=``)
      are downloaded with the ``CIVITAI_TOKEN`` Colab secret if present.
    * Any other URL is downloaded as-is.
    """
    if not is_url(src):
        return src
    m = _HF_BLOB.match(src)
    if m:
        from huggingface_hub import hf_hub_download

        repo, rev, filename = m.groups()
        return hf_hub_download(repo, filename, revision=rev)
    return download_url(src, MODELS_DIR / subdir)


def _civitai_url(url: str) -> str:
    parsed = urlparse(url)
    if "civitai.com" not in parsed.netloc:
        return url
    if "/api/download/models/" in parsed.path:
        dl = url
    else:
        qs = parse_qs(parsed.query)
        version = (qs.get("modelVersionId") or [None])[0]
        if not version:
            raise ValueError(
                "For Civitai model pages, copy the *download* link (right-click Download -> copy link), "
                "e.g. https://civitai.com/api/download/models/123456"
            )
        dl = f"https://civitai.com/api/download/models/{version}"
    token = _colab_secret("CIVITAI_TOKEN")
    if token:
        p = urlparse(dl)
        q = parse_qs(p.query)
        q["token"] = [token]
        dl = urlunparse(p._replace(query=urlencode(q, doseq=True)))
    return dl


def download_url(url: str, dest_dir: Path | str) -> str:
    import requests
    from tqdm.auto import tqdm

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    real = _civitai_url(url)
    with requests.get(real, stream=True, allow_redirects=True, timeout=60) as r:
        if r.status_code in (401, 403):
            raise PermissionError(
                f"Download refused ({r.status_code}). For Civitai add a CIVITAI_TOKEN Colab secret; "
                "for gated HF models add HF_TOKEN and accept the model license."
            )
        r.raise_for_status()
        name = None
        cd = r.headers.get("content-disposition", "")
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
        if m:
            name = m.group(1)
        name = name or Path(urlparse(r.url).path).name or "model.safetensors"
        name = re.sub(r"[^\w.\-]+", "_", name)
        path = dest_dir / name
        if path.exists() and path.stat().st_size > 0:
            print(f"[aurora] Using cached {path}")
            return str(path)
        total = int(r.headers.get("content-length", 0)) or None
        tmp = path.with_suffix(path.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(path)
    if path.suffix not in (".safetensors", ".ckpt", ".pt", ".bin"):
        print(f"[aurora] WARNING: downloaded file {path.name} is not a .safetensors file.")
    return str(path)
