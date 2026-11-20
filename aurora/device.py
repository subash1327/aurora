"""GPU / runtime detection helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    device: str  # "cuda" | "mps" | "cpu"
    name: str
    vram_gb: float
    ram_gb: float
    supports_bf16: bool
    compute_capability: tuple[int, int] | None = None

    @property
    def tier(self) -> str:
        """Rough GPU tier: "cpu", "t4" (<=16 GB), "l4" (<=24 GB), "a100" (<=48 GB), "a100-80"."""
        if self.device != "cuda":
            return "cpu"
        if self.vram_gb <= 16.5:
            return "t4"
        if self.vram_gb <= 24.5:
            return "l4"
        if self.vram_gb <= 48.5:
            return "a100"
        return "a100-80"

    def summary(self) -> str:
        return (
            f"{self.name} | device={self.device} | VRAM={self.vram_gb:.1f} GB | RAM={self.ram_gb:.1f} GB "
            f"| bf16={'yes' if self.supports_bf16 else 'no'} | tier={self.tier}"
        )


def _ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        except Exception:
            return 0.0


def get_device_info() -> DeviceInfo:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cc = (props.major, props.minor)
        return DeviceInfo(
            device="cuda",
            name=props.name,
            vram_gb=props.total_memory / 1024**3,
            ram_gb=_ram_gb(),
            supports_bf16=cc >= (8, 0),  # Ampere+ (A100, L4). T4 (7.5) has no fast bf16.
            compute_capability=cc,
        )
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo("mps", "Apple MPS", 0.0, _ram_gb(), False)
    return DeviceInfo("cpu", "CPU", 0.0, _ram_gb(), False)


def pick_dtype(preference: str = "auto", info: DeviceInfo | None = None):
    """Map "auto" | "bf16" | "fp16" | "fp32" to a torch dtype.

    "auto" picks bf16 on Ampere+ GPUs (L4/A100), fp16 on older GPUs (T4), fp32 on CPU.
    """
    import torch

    info = info or get_device_info()
    pref = (preference or "auto").lower()
    if pref in ("bf16", "bfloat16"):
        return torch.bfloat16
    if pref in ("fp16", "float16", "half"):
        return torch.float16
    if pref in ("fp32", "float32"):
        return torch.float32
    if info.device == "cuda":
        return torch.bfloat16 if info.supports_bf16 else torch.float16
    return torch.float32


def free_memory() -> None:
    """Run GC and empty the CUDA cache."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def vram_report() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "no CUDA device"
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return f"VRAM allocated={alloc:.2f} GB reserved={reserved:.2f} GB peak={peak:.2f} GB"
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def hf_login_from_colab_secret(secret_name: str = "HF_TOKEN") -> bool:
    """Log in to Hugging Face using a Colab secret (🔑 sidebar) or the HF_TOKEN env var."""
    token = os.environ.get(secret_name)
    if not token and in_colab():
        try:
            from google.colab import userdata

            token = userdata.get(secret_name)
        except Exception:
            token = None
    if not token:
        print(f"[aurora] No {secret_name} found; public models only. Add it in Colab's 🔑 Secrets panel for gated models.")
        return False
    os.environ["HF_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        print(f"[aurora] huggingface_hub login failed: {exc}")
        return False
    print("[aurora] Logged in to Hugging Face.")
    return True
