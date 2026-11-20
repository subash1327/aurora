"""Wan 2.1 / 2.2 video generation helpers (text-to-video and image-to-video) via diffusers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .device import DeviceInfo, free_memory, get_device_info, pick_dtype
from .memory import apply_memory_settings
from .safety import assert_safe

# UMT5-XXL encoder size in GB at 16-bit. Same for every Wan checkpoint.
_TEXT_ENCODER_GB = 11.4

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)


@dataclass(frozen=True)
class WanModel:
    key: str
    repo: str
    modes: tuple[str, ...]  # "t2v" and/or "i2v"
    transformer_gb: float  # 16-bit weights of the denoiser(s), summed
    largest_gb: float  # largest single denoiser at 16-bit
    height: int
    width: int
    num_frames: int
    fps: int
    steps: int = 30
    guidance: float = 5.0
    flow_shift: float | None = None  # None = keep the scheduler shipped with the checkpoint
    image_encoder: bool = False  # Wan 2.1 I2V uses a CLIP image encoder
    min_tier: str = "t4"
    notes: str = ""
    extra: dict = field(default_factory=dict)


WAN_MODELS: dict[str, WanModel] = {
    m.key: m
    for m in (
        WanModel(
            "Wan2.1-T2V-1.3B",
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            ("t2v",),
            2.8,
            2.8,
            480,
            832,
            81,
            16,
            steps=30,
            guidance=5.0,
            flow_shift=3.0,
            min_tier="t4",
            notes="Free Colab T4 default. 480p, ~5 s clips.",
        ),
        WanModel(
            "Wan2.2-TI2V-5B",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            ("t2v", "i2v"),
            10.0,
            10.0,
            704,
            1280,
            121,
            24,
            steps=40,
            guidance=5.0,
            min_tier="t4",
            notes="Text+image to video at 720p/24fps. On T4 it is quantized (4-bit) and slow; L4 recommended.",
        ),
        WanModel(
            "Wan2.1-T2V-14B",
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            ("t2v",),
            28.6,
            28.6,
            720,
            1280,
            81,
            16,
            steps=40,
            guidance=5.0,
            flow_shift=5.0,
            min_tier="a100",
            notes="High quality T2V. A100 40/80 GB.",
        ),
        WanModel(
            "Wan2.1-I2V-14B-480P",
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            ("i2v",),
            32.8,
            32.8,
            480,
            832,
            81,
            16,
            steps=40,
            guidance=5.0,
            flow_shift=3.0,
            image_encoder=True,
            min_tier="a100",
            notes="Image-to-video 480p. A100.",
        ),
        WanModel(
            "Wan2.1-I2V-14B-720P",
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            ("i2v",),
            32.8,
            32.8,
            720,
            1280,
            81,
            16,
            steps=40,
            guidance=5.0,
            flow_shift=5.0,
            image_encoder=True,
            min_tier="a100",
            notes="Image-to-video 720p. A100.",
        ),
        WanModel(
            "Wan2.2-T2V-A14B",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            ("t2v",),
            57.2,
            28.6,
            720,
            1280,
            81,
            16,
            steps=40,
            guidance=4.0,
            min_tier="a100",
            notes="MoE (two 14B experts). A100 80 GB, or A100 40 GB with offload/quantization.",
        ),
        WanModel(
            "Wan2.2-I2V-A14B",
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            ("i2v",),
            57.2,
            28.6,
            720,
            1280,
            81,
            16,
            steps=40,
            guidance=3.5,
            min_tier="a100",
            notes="MoE image-to-video. A100 80 GB, or A100 40 GB with offload/quantization.",
        ),
    )
}

_TIER_ORDER = {"cpu": 0, "t4": 1, "l4": 2, "a100": 3, "a100-80": 4}


def list_models(mode: str | None = None) -> list[str]:
    return [k for k, m in WAN_MODELS.items() if mode is None or mode in m.modes]


def default_model(mode: str = "t2v", info: DeviceInfo | None = None) -> str:
    info = info or get_device_info()
    big = _TIER_ORDER.get(info.tier, 0) >= _TIER_ORDER["a100"]
    if mode == "i2v":
        return "Wan2.1-I2V-14B-480P" if big else "Wan2.2-TI2V-5B"
    return "Wan2.1-T2V-14B" if big else "Wan2.1-T2V-1.3B"


def snap_frames(num_frames: int) -> int:
    """Wan needs num_frames = 4k + 1."""
    num_frames = max(5, int(num_frames))
    return 4 * round((num_frames - 1) / 4) + 1


def snap_size(height: int, width: int, multiple: int = 16) -> tuple[int, int]:
    return max(multiple, int(height) // multiple * multiple), max(multiple, int(width) // multiple * multiple)


def resolve_quantization(quantize: str, spec: WanModel, info: DeviceInfo) -> dict[str, str]:
    """Return {component: "8bit"|"4bit"} for bitsandbytes quantization.

    "auto": on <=16 GB GPUs quantize the 11 GB UMT5 text encoder to 8-bit, and the denoiser to 4-bit (NF4)
    when it would not fit next to it. On 24 GB GPUs only checkpoints >= 14B get a 8-bit text encoder.
    """
    q = (quantize or "auto").lower()
    if q == "none" or info.device != "cuda":
        return {}
    if q in ("8bit", "4bit"):
        return {"text_encoder": q, "transformer": q, **({"transformer_2": q} if spec.largest_gb < spec.transformer_gb else {})}
    if q == "text_encoder_8bit":
        return {"text_encoder": "8bit"}
    if q != "auto":
        raise ValueError("quantize must be auto|none|8bit|4bit|text_encoder_8bit")
    plan: dict[str, str] = {}
    if info.tier == "t4":
        plan["text_encoder"] = "8bit"
        if spec.largest_gb > 4:
            plan["transformer"] = "4bit"
            if spec.largest_gb < spec.transformer_gb:
                plan["transformer_2"] = "4bit"
    elif info.tier == "l4" and spec.largest_gb > 12:
        plan["text_encoder"] = "8bit"
        plan["transformer"] = "4bit"
        if spec.largest_gb < spec.transformer_gb:
            plan["transformer_2"] = "4bit"
    return plan


def _quant_config(plan: dict[str, str], compute_dtype):
    if not plan:
        return None
    from diffusers.quantizers import PipelineQuantizationConfig

    try:
        from diffusers import BitsAndBytesConfig as DiffusersBnb
        from transformers import BitsAndBytesConfig as TransformersBnb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Quantization needs `pip install bitsandbytes`.") from exc

    def cfg(kind: str, transformers_side: bool):
        cls = TransformersBnb if transformers_side else DiffusersBnb
        if kind == "8bit":
            return cls(load_in_8bit=True)
        return cls(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype)

    mapping = {name: cfg(kind, transformers_side=(name == "text_encoder")) for name, kind in plan.items()}
    return PipelineQuantizationConfig(quant_mapping=mapping)


def _estimate_gb(spec: WanModel, plan: dict[str, str], dtype_bytes: int) -> tuple[float, float]:
    scale = {"8bit": 0.5, "4bit": 0.3, None: dtype_bytes / 2}
    te = _TEXT_ENCODER_GB * scale[plan.get("text_encoder")]
    tr = spec.transformer_gb * scale[plan.get("transformer")]
    largest = spec.largest_gb * scale[plan.get("transformer")]
    vae = 1.0
    return te + tr + vae, max(te, largest)


class WanGenerator:
    """Load a Wan checkpoint once, generate many clips."""

    def __init__(
        self,
        model: str = "auto",
        mode: str = "t2v",
        dtype: str = "auto",
        offload: str = "auto",
        quantize: str = "auto",
        vae_tiling: bool = True,
        info: DeviceInfo | None = None,
    ):
        import torch
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanPipeline

        self.info = info or get_device_info()
        mode = mode.lower()
        if mode not in ("t2v", "i2v"):
            raise ValueError("mode must be 't2v' or 'i2v'")
        if model == "auto":
            model = default_model(mode, self.info)
        if model not in WAN_MODELS:
            raise ValueError(f"Unknown model {model!r}. Choose from: {', '.join(WAN_MODELS)}")
        self.spec = spec = WAN_MODELS[model]
        if mode not in spec.modes:
            raise ValueError(f"{model} does not support {mode}; it supports {spec.modes}")
        if _TIER_ORDER.get(self.info.tier, 0) < _TIER_ORDER[spec.min_tier]:
            print(f"[aurora] WARNING: {model} is meant for {spec.min_tier}+ GPUs; you have {self.info.summary()}. Expect OOM.")
        self.mode = mode
        self.dtype = pick_dtype(dtype, self.info)
        self.quant_plan = resolve_quantization(quantize, spec, self.info)

        print(f"[aurora] Loading {spec.repo} ({mode}) dtype={self.dtype} quant={self.quant_plan or 'none'}")
        # VAE in fp32 for decode quality (recommended by diffusers docs); it is small.
        vae = AutoencoderKLWan.from_pretrained(spec.repo, subfolder="vae", torch_dtype=torch.float32)
        kwargs = {"vae": vae, "torch_dtype": self.dtype}
        if spec.image_encoder and mode == "i2v":
            from transformers import CLIPVisionModel

            kwargs["image_encoder"] = CLIPVisionModel.from_pretrained(
                spec.repo, subfolder="image_encoder", torch_dtype=torch.float32
            )
        qc = _quant_config(self.quant_plan, self.dtype)
        if qc is not None:
            kwargs["quantization_config"] = qc
        cls = WanPipeline if mode == "t2v" else WanImageToVideoPipeline
        self.pipe = cls.from_pretrained(spec.repo, **kwargs)
        if spec.flow_shift is not None:
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=spec.flow_shift)
        total, largest = _estimate_gb(spec, self.quant_plan, 2 if self.dtype != torch.float32 else 4)
        self.offload = apply_memory_settings(
            self.pipe, offload=offload, total_gb=total, largest_gb=largest, vae_tiling=vae_tiling, info=self.info
        )

    # ------------------------------------------------------------------ helpers
    @property
    def spatial_multiple(self) -> int:
        p = self.pipe
        try:
            return int(p.vae_scale_factor_spatial * p.transformer.config.patch_size[1])
        except Exception:
            return 16

    def fit_image(self, image, max_area: int | None = None):
        """Resize an input image to the model's area keeping aspect ratio; returns (image, h, w)."""
        max_area = max_area or self.spec.height * self.spec.width
        aspect = image.height / image.width
        mod = self.spatial_multiple
        h = max(mod, round(math.sqrt(max_area * aspect)) // mod * mod)
        w = max(mod, round(math.sqrt(max_area / aspect)) // mod * mod)
        return image.convert("RGB").resize((w, h)), h, w

    # ------------------------------------------------------------------ generate
    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = DEFAULT_NEGATIVE_PROMPT,
        image=None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int = -1,
        **extra,
    ):
        """Return (frames, used_params). ``image`` (PIL / path / URL) is required for i2v."""
        import random

        import torch
        from diffusers.utils import load_image

        assert_safe(prompt)
        spec = self.spec
        seed = random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)
        num_frames = snap_frames(num_frames or spec.num_frames)
        params = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_frames=num_frames,
            num_inference_steps=int(steps or spec.steps),
            guidance_scale=float(guidance if guidance is not None else spec.guidance),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            **extra,
        )
        if self.mode == "i2v":
            if image is None:
                raise ValueError("image-to-video needs an input image")
            if isinstance(image, str):
                image = load_image(image)
            area = (height or spec.height) * (width or spec.width)
            image, h, w = self.fit_image(image, max_area=area)
            params.update(image=image, height=h, width=w)
        else:
            h, w = snap_size(height or spec.height, width or spec.width, self.spatial_multiple)
            params.update(height=h, width=w)

        free_memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        frames = self.pipe(**params).frames[0]
        used = {k: v for k, v in params.items() if k not in ("generator", "image")}
        used.update(seed=seed, model=spec.key, repo=spec.repo, mode=self.mode, dtype=str(self.dtype), quant=self.quant_plan)
        return frames, used

    def unload(self):
        del self.pipe
        free_memory()
