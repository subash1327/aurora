"""SDXL (and optional Flux) image generation without a safety checker, plus Aurora's prompt filter."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

from .device import DeviceInfo, free_memory, get_device_info, pick_dtype
from .downloads import is_url, resolve_local
from .memory import apply_memory_settings
from .safety import PromptRefused, assert_safe, check_prompt

DEFAULT_SDXL_NEGATIVE = (
    "lowres, worst quality, low quality, bad anatomy, bad hands, extra fingers, missing fingers, "
    "deformed, blurry, jpeg artifacts, watermark, signature, text"
)


@dataclass(frozen=True)
class ImageModel:
    key: str
    repo: str
    family: str  # "sdxl" | "flux"
    steps: int
    guidance: float
    width: int = 1024
    height: int = 1024
    scheduler: str = "DPM++ 2M Karras"
    clip_skip: int | None = None
    notes: str = ""


IMAGE_MODELS: dict[str, ImageModel] = {
    m.key: m
    for m in (
        ImageModel(
            "Juggernaut XL v9 (photoreal)",
            "RunDiffusion/Juggernaut-XL-v9",
            "sdxl",
            30,
            5.0,
            width=832,
            height=1216,
            notes="Popular photoreal SDXL fine-tune; strong anatomy.",
        ),
        ImageModel(
            "RealVisXL V5.0 (photoreal)",
            "SG161222/RealVisXL_V5.0",
            "sdxl",
            30,
            5.0,
            width=832,
            height=1216,
            notes="Photoreal SDXL fine-tune.",
        ),
        ImageModel(
            "Animagine XL 4.0 (anime)",
            "cagliostrolab/animagine-xl-4.0",
            "sdxl",
            28,
            5.0,
            width=832,
            height=1216,
            scheduler="Euler a",
            notes="Anime, Danbooru-style tags.",
        ),
        ImageModel(
            "Illustrious XL v0.1 (anime)",
            "OnomaAIResearch/Illustrious-xl-early-release-v0",
            "sdxl",
            28,
            6.0,
            width=832,
            height=1216,
            scheduler="Euler a",
            clip_skip=2,
            notes="Anime/illustration base.",
        ),
        ImageModel(
            "DreamShaper XL v2 Turbo (fast)",
            "Lykon/dreamshaper-xl-v2-turbo",
            "sdxl",
            8,
            2.0,
            width=896,
            height=1152,
            scheduler="DPM++ SDE Karras",
            notes="Turbo: 6-8 steps, CFG ~2.",
        ),
        ImageModel(
            "SDXL Base 1.0",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "sdxl",
            30,
            7.0,
            notes="Official base model (more conservative training data).",
        ),
        ImageModel(
            "FLUX.1 schnell (L4/A100)",
            "black-forest-labs/FLUX.1-schnell",
            "flux",
            4,
            0.0,
            notes="Gated on HF: accept license + HF_TOKEN. 4 steps, no CFG.",
        ),
        ImageModel(
            "FLUX.1 dev (L4/A100)",
            "black-forest-labs/FLUX.1-dev",
            "flux",
            28,
            3.5,
            notes="Gated, non-commercial license. Needs HF_TOKEN.",
        ),
    )
}

SCHEDULERS = ("DPM++ 2M Karras", "DPM++ 2M", "DPM++ SDE Karras", "Euler a", "Euler", "UniPC", "default")

SDXL_FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"


def set_scheduler(pipe, name: str):
    from diffusers import (
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        UniPCMultistepScheduler,
    )

    cfg = pipe.scheduler.config
    if name == "default" or not name:
        return pipe
    if name == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(cfg, use_karras_sigmas=True)
    elif name == "DPM++ 2M":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(cfg)
    elif name == "DPM++ SDE Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(cfg, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(cfg)
    elif name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(cfg)
    elif name == "UniPC":
        pipe.scheduler = UniPCMultistepScheduler.from_config(cfg)
    else:
        raise ValueError(f"Unknown scheduler {name!r}; choose from {SCHEDULERS}")
    return pipe


def _check_source_name(src: str) -> None:
    """Refuse checkpoints/LoRAs whose name advertises minor content (e.g. '...loli...')."""
    name = os.path.basename(str(src)).replace("_", " ").replace("-", " ")
    res = check_prompt(name)
    if not res.allowed:
        raise PromptRefused(f"Refusing to load {src!r}: name matches blocked terms {res.matches}.")


class ImageGenerator:
    """SDXL / Flux text-to-image and image-to-image. No safety checker is loaded.

    ``model`` is a key of :data:`IMAGE_MODELS`, a diffusers-format HF repo id, or a single-file
    checkpoint (local path, HF file URL or Civitai download URL). ``family`` is needed for custom
    sources ("sdxl" by default).
    """

    def __init__(
        self,
        model: str = "Juggernaut XL v9 (photoreal)",
        family: str | None = None,
        dtype: str = "auto",
        offload: str = "auto",
        scheduler: str | None = None,
        fp16_vae_fix: bool = True,
        info: DeviceInfo | None = None,
    ):
        import torch

        self.info = info or get_device_info()
        spec = IMAGE_MODELS.get(model)
        self.spec = spec
        self.family = (spec.family if spec else (family or "sdxl")).lower()
        self.source = spec.repo if spec else model
        self.loras: dict[str, float] = {}
        if spec is None:
            _check_source_name(self.source)

        if self.family == "flux" and not self.info.supports_bf16 and dtype == "auto":
            print("[aurora] WARNING: Flux needs bf16 (L4/A100). On a T4 it will be very slow or produce black images.")
            dtype = "bf16"
        self.dtype = pick_dtype(dtype, self.info)

        if self.family == "sdxl":
            self.pipe = self._load_sdxl(fp16_vae_fix)
            sched = scheduler or (spec.scheduler if spec else "DPM++ 2M Karras")
            set_scheduler(self.pipe, sched)
            total, largest = 7.5 if self.dtype != torch.float32 else 14.0, 5.5
        elif self.family == "flux":
            self.pipe = self._load_flux()
            total, largest = (34.0, 24.0) if not self._flux_quantized else (14.0, 7.0)
        else:
            raise ValueError("family must be 'sdxl' or 'flux'")
        self.offload = apply_memory_settings(self.pipe, offload=offload, total_gb=total, largest_gb=largest, info=self.info)
        self._img2img = None

    # ------------------------------------------------------------------ loading
    def _load_sdxl(self, fp16_vae_fix: bool):
        import torch
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline

        kwargs = {"torch_dtype": self.dtype, "use_safetensors": True}
        if fp16_vae_fix and self.dtype == torch.float16:
            # The stock SDXL VAE overflows (NaN / black images) in fp16; this VAE is fp16-safe.
            kwargs["vae"] = AutoencoderKL.from_pretrained(SDXL_FP16_VAE, torch_dtype=torch.float16)
        src = self.source
        if is_url(src) or src.endswith((".safetensors", ".ckpt")):
            path = resolve_local(src, "checkpoints")
            print(f"[aurora] Loading single-file SDXL checkpoint {path}")
            kwargs.pop("use_safetensors", None)
            return StableDiffusionXLPipeline.from_single_file(path, **kwargs)
        print(f"[aurora] Loading SDXL {src} dtype={self.dtype}")
        if self.dtype == torch.float16:
            try:
                return StableDiffusionXLPipeline.from_pretrained(src, variant="fp16", **kwargs)
            except (OSError, ValueError):
                pass  # repo has no fp16 variant files
        return StableDiffusionXLPipeline.from_pretrained(src, **kwargs)

    def _load_flux(self):
        import torch
        from diffusers import FluxPipeline

        self._flux_quantized = self.info.vram_gb < 40
        kwargs = {"torch_dtype": self.dtype}
        if self._flux_quantized:
            from diffusers.quantizers import PipelineQuantizationConfig

            kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
                components_to_quantize=["transformer", "text_encoder_2"],
            )
        src = self.source
        if is_url(src) or src.endswith(".safetensors"):
            from diffusers import FluxTransformer2DModel

            path = resolve_local(src, "checkpoints")
            base = "black-forest-labs/FLUX.1-dev"
            kwargs.pop("quantization_config", None)
            transformer = FluxTransformer2DModel.from_single_file(path, torch_dtype=self.dtype)
            return FluxPipeline.from_pretrained(base, transformer=transformer, **kwargs)
        print(f"[aurora] Loading Flux {src} dtype={self.dtype} nf4={self._flux_quantized}")
        return FluxPipeline.from_pretrained(src, **kwargs)

    # ------------------------------------------------------------------ LoRA
    def add_lora(self, source: str, weight: float = 0.8, name: str | None = None, weight_name: str | None = None):
        """Load a LoRA from an HF repo id, HF/Civitai URL or local .safetensors file."""
        _check_source_name(source)
        name = name or f"lora{len(self.loras) + 1}"
        path = resolve_local(source, "loras")
        kwargs = {"adapter_name": name}
        if weight_name:
            kwargs["weight_name"] = weight_name
        self.pipe.load_lora_weights(path, **kwargs)
        self.loras[name] = float(weight)
        self.pipe.set_adapters(list(self.loras), adapter_weights=list(self.loras.values()))
        print(f"[aurora] LoRA {name} <- {source} (weight {weight})")
        return name

    def set_lora_weight(self, name: str, weight: float):
        self.loras[name] = float(weight)
        self.pipe.set_adapters(list(self.loras), adapter_weights=list(self.loras.values()))

    def clear_loras(self):
        if self.loras:
            self.pipe.unload_lora_weights()
        self.loras.clear()

    # ------------------------------------------------------------------ generate
    def _img2img_pipe(self):
        if self._img2img is None:
            if self.family == "sdxl":
                from diffusers import StableDiffusionXLImg2ImgPipeline as Cls
            else:
                from diffusers import FluxImg2ImgPipeline as Cls
            self._img2img = Cls.from_pipe(self.pipe)
        return self._img2img

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = DEFAULT_SDXL_NEGATIVE,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int = -1,
        batch_size: int = 1,
        init_image=None,
        strength: float = 0.6,
        clip_skip: int | None = None,
        **extra,
    ):
        """txt2img (``init_image=None``) or img2img. Returns (images, list_of_param_dicts)."""
        import torch
        from diffusers.utils import load_image

        assert_safe(prompt)
        spec = self.spec
        steps = int(steps or (spec.steps if spec else (28 if self.family == "flux" else 30)))
        if guidance is None:
            guidance = spec.guidance if spec else (3.5 if self.family == "flux" else 6.0)
        width = int(width or (spec.width if spec else 1024)) // 8 * 8
        height = int(height or (spec.height if spec else 1024)) // 8 * 8
        base_seed = random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)
        seeds = [base_seed + i for i in range(max(1, int(batch_size)))]
        gens = [torch.Generator(device="cpu").manual_seed(s) for s in seeds]

        params = dict(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=float(guidance),
            generator=gens,
            num_images_per_prompt=len(seeds),
            **extra,
        )
        if self.family == "sdxl":
            params["negative_prompt"] = negative_prompt or None
            cs = clip_skip if clip_skip is not None else (spec.clip_skip if spec else None)
            if cs:
                params["clip_skip"] = int(cs) - 1 if int(cs) > 1 else None  # diffusers counts layers skipped
        if init_image is not None:
            if isinstance(init_image, str):
                init_image = load_image(init_image)
            init_image = init_image.convert("RGB").resize((width, height))
            pipe = self._img2img_pipe()
            params.update(image=init_image, strength=float(strength))
            if self.family == "flux":
                params.update(width=width, height=height)
        else:
            pipe = self.pipe
            params.update(width=width, height=height)

        free_memory()
        images = pipe(**params).images
        meta = []
        for s in seeds:
            m = {k: v for k, v in params.items() if k not in ("generator", "image", "num_images_per_prompt")}
            m.update(
                seed=s,
                model=self.source,
                family=self.family,
                loras=dict(self.loras),
                mode="img2img" if init_image is not None else "txt2img",
                dtype=str(self.dtype),
            )
            meta.append(m)
        return images, meta

    def unload(self):
        self._img2img = None
        del self.pipe
        free_memory()
