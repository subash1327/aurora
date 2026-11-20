"""CPU-only checks of model registries and memory/quantization planning (no downloads)."""

import pytest

from aurora.device import DeviceInfo
from aurora.memory import resolve_offload
from aurora.video import WAN_MODELS, default_model, resolve_quantization, snap_frames

T4 = DeviceInfo("cuda", "Tesla T4", 14.7, 12.7, False, (7, 5))
L4 = DeviceInfo("cuda", "NVIDIA L4", 22.5, 53.0, True, (8, 9))
A100 = DeviceInfo("cuda", "NVIDIA A100-SXM4-40GB", 39.6, 83.0, True, (8, 0))


def test_snap_frames():
    assert snap_frames(81) == 81
    assert snap_frames(80) == 81
    assert snap_frames(50) == 49
    assert (snap_frames(3) - 1) % 4 == 0


def test_defaults_per_tier():
    assert default_model("t2v", T4) == "Wan2.1-T2V-1.3B"
    assert default_model("i2v", T4) == "Wan2.2-TI2V-5B"
    assert default_model("t2v", A100) == "Wan2.1-T2V-14B"
    for mode in ("t2v", "i2v"):
        for info in (T4, L4, A100):
            assert mode in WAN_MODELS[default_model(mode, info)].modes


def test_quant_plan_t4():
    assert resolve_quantization("auto", WAN_MODELS["Wan2.1-T2V-1.3B"], T4) == {"text_encoder": "8bit"}
    plan = resolve_quantization("auto", WAN_MODELS["Wan2.2-TI2V-5B"], T4)
    assert plan == {"text_encoder": "8bit", "transformer": "4bit"}
    moe = resolve_quantization("auto", WAN_MODELS["Wan2.2-T2V-A14B"], T4)
    assert moe["transformer_2"] == "4bit"
    assert resolve_quantization("auto", WAN_MODELS["Wan2.1-T2V-14B"], A100) == {}


def test_offload():
    assert resolve_offload("auto", 6.5, 3.0, T4) == "none"
    assert resolve_offload("auto", 20.0, 10.0, L4) == "model"
    assert resolve_offload("auto", 40.0, 28.6, L4) == "sequential"
    with pytest.raises(ValueError):
        resolve_offload("bogus", 1, 1, T4)


def test_diffusers_api_surface():
    diffusers = pytest.importorskip("diffusers")
    for name in (
        "WanPipeline",
        "WanImageToVideoPipeline",
        "AutoencoderKLWan",
        "UniPCMultistepScheduler",
        "StableDiffusionXLPipeline",
        "StableDiffusionXLImg2ImgPipeline",
        "FluxPipeline",
        "FluxImg2ImgPipeline",
        "BitsAndBytesConfig",
    ):
        assert hasattr(diffusers, name), name
