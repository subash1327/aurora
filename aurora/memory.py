"""Memory-saving settings for diffusers pipelines on small GPUs."""

from __future__ import annotations

from .device import DeviceInfo, get_device_info

OFFLOAD_MODES = ("auto", "none", "model", "sequential")


def resolve_offload(
    mode: str,
    total_gb: float,
    largest_gb: float | None = None,
    info: DeviceInfo | None = None,
    activation_headroom_gb: float = 4.0,
) -> str:
    """Pick an offload strategy.

    ``total_gb``: estimated weights of all components on the GPU at once.
    ``largest_gb``: weights of the largest single component (what model offload needs resident).
    """
    mode = (mode or "auto").lower()
    if mode not in OFFLOAD_MODES:
        raise ValueError(f"offload must be one of {OFFLOAD_MODES}, got {mode!r}")
    if mode != "auto":
        return mode
    info = info or get_device_info()
    if info.device != "cuda":
        return "none"
    largest_gb = total_gb if largest_gb is None else largest_gb
    if info.vram_gb - total_gb >= activation_headroom_gb:
        return "none"
    if info.vram_gb - largest_gb >= activation_headroom_gb - 1:
        return "model"
    return "sequential"


def apply_memory_settings(
    pipe,
    offload: str = "auto",
    total_gb: float = 8.0,
    largest_gb: float | None = None,
    vae_tiling: bool = True,
    vae_slicing: bool = True,
    attention_slicing: bool = False,
    info: DeviceInfo | None = None,
    verbose: bool = True,
):
    """Apply offloading + VAE tiling/slicing to ``pipe`` and move it to the right device."""
    info = info or get_device_info()
    mode = resolve_offload(offload, total_gb, largest_gb, info)
    if info.device == "cuda":
        if mode == "model":
            pipe.enable_model_cpu_offload()
        elif mode == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cuda")
    else:
        pipe.to(info.device)

    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if vae_tiling and hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
            except Exception as exc:  # pragma: no cover - model specific
                print(f"[aurora] VAE tiling not available: {exc}")
        if vae_slicing and hasattr(vae, "enable_slicing"):
            try:
                vae.enable_slicing()
            except Exception:
                pass
    if attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if verbose:
        print(f"[aurora] offload={mode} vae_tiling={vae_tiling} vae_slicing={vae_slicing} on {info.name}")
    return mode
