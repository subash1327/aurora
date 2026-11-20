"""Regenerate notebooks/*.ipynb from the cell sources below: `python tools/build_notebooks.py`."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
REPO = "subash1327/aurora"


def colab_badge(nb: str) -> str:
    return (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{nb})"
    )


INSTALL = """#@title 1. Install Aurora + dependencies (≈2 min)
AURORA_REF = "main"  #@param {type:"string"}
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU: Runtime > Change runtime type > T4 GPU"
!pip install -q -U -r https://raw.githubusercontent.com/subash1327/aurora/{AURORA_REF}/requirements.txt
!pip install -q -U --no-deps --force-reinstall "git+https://github.com/subash1327/aurora@{AURORA_REF}"
print("Installed. If Colab asks to restart the session, restart and continue from step 2.")"""

LOGIN = """#@title 2. Hugging Face login (Colab secret `HF_TOKEN`) + GPU check
#@markdown Add `HF_TOKEN` in the 🔑 **Secrets** sidebar and enable notebook access. Needed for gated models (e.g. FLUX).
from aurora.device import get_device_info, hf_login_from_colab_secret

hf_login_from_colab_secret("HF_TOKEN")
INFO = get_device_info()
print(INFO.summary())
if INFO.device != "cuda":
    print("⚠️  No GPU detected: Runtime > Change runtime type > T4 GPU (or L4 / A100).")"""

# ----------------------------------------------------------------------------- video notebook
VIDEO_MODELS = [
    "auto",
    "Wan2.1-T2V-1.3B",
    "Wan2.2-TI2V-5B",
    "Wan2.1-T2V-14B",
    "Wan2.1-I2V-14B-480P",
    "Wan2.1-I2V-14B-720P",
    "Wan2.2-T2V-A14B",
    "Wan2.2-I2V-A14B",
]

video_cells = [
    new_markdown_cell(f"""# 🌌 Aurora · Wan video generation

{colab_badge("aurora_video_wan.ipynb")}

Text-to-video and image-to-video with [Wan 2.1 / 2.2](https://huggingface.co/Wan-AI) through 🤗 diffusers.

| Model | Mode | Runs on |
|---|---|---|
| **Wan2.1-T2V-1.3B** (default) | text → video, 480p | free **T4** |
| **Wan2.2-TI2V-5B** | text/image → video, 720p 24 fps | T4 (4-bit, slow) · **L4** recommended |
| Wan2.1-T2V-14B, Wan2.1-I2V-14B-480P/720P | text / image → video | **A100** |
| Wan2.2-T2V-A14B, Wan2.2-I2V-A14B (MoE) | text / image → video | A100 80 GB (40 GB with offload) |

**Steps:** run 1 → 2 → 3 (load model) → 4 (generate). Re-run 4 as often as you like.

> **Content policy:** no sexual content involving minors (prompts with child/teen terms are refused automatically) and no sexual or non-consensual content of real, identifiable people. See the README."""),
    new_code_cell(INSTALL),
    new_code_cell(LOGIN),
    new_code_cell(f"""#@title 3. Load a Wan model
MODEL = "auto"  #@param {json.dumps(VIDEO_MODELS)}
MODE = "t2v"  #@param ["t2v", "i2v"]
#@markdown `auto` = Wan2.1-T2V-1.3B (t2v) / Wan2.2-TI2V-5B (i2v) on T4/L4, 14B models on A100.
DTYPE = "auto"  #@param ["auto", "fp16", "bf16", "fp32"]
#@markdown `auto`: bf16 on L4/A100, fp16 on T4 (no fast bf16).
OFFLOAD = "auto"  #@param ["auto", "none", "model", "sequential"]
QUANTIZE = "auto"  #@param ["auto", "none", "text_encoder_8bit", "8bit", "4bit"]
#@markdown `auto` on T4: 8-bit text encoder (+ 4-bit transformer for 5B+). `sequential` offload = lowest VRAM, slowest.
VAE_TILING = True  #@param {{type:"boolean"}}

from aurora.device import free_memory
from aurora.video import WanGenerator

_key = (MODEL, MODE, DTYPE, OFFLOAD, QUANTIZE, VAE_TILING)
if globals().get("GEN") is None or globals().get("_GEN_KEY") != _key:
    if globals().get("GEN") is not None:
        GEN.unload()
        GEN = None
        free_memory()
    GEN = WanGenerator(model=MODEL, mode=MODE, dtype=DTYPE, offload=OFFLOAD, quantize=QUANTIZE, vae_tiling=VAE_TILING)
    _GEN_KEY = _key
s = GEN.spec
print(f"Ready: {{s.key}} [{{GEN.mode}}] defaults {{s.height}}x{{s.width}}, {{s.num_frames}} frames @ {{s.fps}} fps, "
      f"{{s.steps}} steps, guidance {{s.guidance}}. {{s.notes}}")"""),
    new_code_cell("""#@title 4. Generate a video
PROMPT = "A cinematic drone shot gliding over snowy mountains at sunrise, golden light, volumetric fog, ultra detailed"  #@param {type:"string"}
NEGATIVE_PROMPT = "default"  #@param {type:"string"}
#@markdown `default` = Wan's recommended negative prompt. Leave blank for none.
RESOLUTION = "model default"  #@param ["model default", "480x832 (landscape)", "832x480 (portrait)", "480x480", "704x1280 (landscape 720p)", "1280x704 (portrait 720p)", "720x1280 (landscape)", "1280x720 (portrait)"]
NUM_FRAMES = 81  #@param {type:"slider", min:17, max:121, step:4}
#@markdown Frames must be 4k+1 (snapped automatically). Wan2.1: 81 @ 16 fps ≈ 5 s. Wan2.2-5B: 121 @ 24 fps ≈ 5 s.
FPS = 0  #@param {type:"slider", min:0, max:30, step:1}
#@markdown `0` = model default (16 for Wan 2.1, 24 for Wan2.2-TI2V-5B).
STEPS = 30  #@param {type:"slider", min:4, max:60, step:1}
GUIDANCE = 5.0  #@param {type:"slider", min:1.0, max:10.0, step:0.5}
SEED = -1  #@param {type:"integer"}
#@markdown `-1` = random seed.
INPUT_IMAGE = ""  #@param {type:"string"}
#@markdown **i2v only:** image URL or a path in `/content`. Leave blank to upload a file.
SAVE_TO_DRIVE = False  #@param {type:"boolean"}

import time

from aurora.device import vram_report
from aurora.outputs import copy_to_drive, save_video, show_video
from aurora.video import DEFAULT_NEGATIVE_PROMPT

neg = DEFAULT_NEGATIVE_PROMPT if NEGATIVE_PROMPT.strip() == "default" else NEGATIVE_PROMPT
h = w = None
if RESOLUTION != "model default":
    h, w = (int(x) for x in RESOLUTION.split()[0].split("x"))

image = None
if GEN.mode == "i2v":
    if INPUT_IMAGE.strip():
        image = INPUT_IMAGE.strip()
    else:
        from google.colab import files
        from PIL import Image
        up = files.upload()
        image = Image.open(next(iter(up)))

t0 = time.time()
frames, params = GEN.generate(PROMPT, negative_prompt=neg, image=image, height=h, width=w,
                              num_frames=NUM_FRAMES, steps=STEPS, guidance=GUIDANCE, seed=SEED)
fps = FPS or GEN.spec.fps
params["fps"] = fps
path = save_video(frames, PROMPT, fps=fps, seed=params["seed"], metadata=params)
print(f"Done in {time.time() - t0:.0f}s | seed={params['seed']} | {params['height']}x{params['width']} x {params['num_frames']} frames | {vram_report()}")
print(f"Saved: {path}")
show_video(path)
if SAVE_TO_DRIVE:
    copy_to_drive(path, "Aurora/videos")"""),
    new_code_cell("""#@title (Optional) Download the last video
from google.colab import files

files.download(str(path))"""),
    new_markdown_cell("""## Tips & troubleshooting

* **CUDA out of memory**: lower `NUM_FRAMES` (e.g. 49) or resolution, set `OFFLOAD = "sequential"`, or `QUANTIZE = "4bit"`. Re-run step 3 after changing load settings. *Runtime > Restart session* clears fragmented VRAM.
* **Session crashed / RAM full while loading** (free Colab has ~12.7 GB system RAM): keep `QUANTIZE = "auto"` (8-bit text encoder), use the 1.3B model, or switch to a High-RAM / L4 runtime.
* **Black or noisy video on T4**: fp16 can overflow on some checkpoints. Try `DTYPE = "fp32"` with `OFFLOAD = "sequential"` (slow), or use an L4/A100 with bf16.
* **Slow**: the 1.3B model on a T4 takes several minutes per 5 s clip. Fewer steps (20-25) is a good trade-off.
* Wan 2.2-TI2V-5B is trained at 720p (704x1280); lower resolutions work but look softer.
"""),
]

# ----------------------------------------------------------------------------- image notebook
IMAGE_MODELS = [
    "Juggernaut XL v9 (photoreal)",
    "RealVisXL V5.0 (photoreal)",
    "Animagine XL 4.0 (anime)",
    "Illustrious XL v0.1 (anime)",
    "DreamShaper XL v2 Turbo (fast)",
    "SDXL Base 1.0",
    "FLUX.1 schnell (L4/A100)",
    "FLUX.1 dev (L4/A100)",
    "Custom (CUSTOM_SOURCE below)",
]
SCHEDULERS = ["model default", "DPM++ 2M Karras", "DPM++ 2M", "DPM++ SDE Karras", "Euler a", "Euler", "UniPC"]

image_cells = [
    new_markdown_cell(f"""# 🌌 Aurora · Image generation (SDXL / Flux, NSFW-capable)

{colab_badge("aurora_image.ipynb")}

SDXL text-to-image and image-to-image with 🤗 diffusers, **no safety checker**, so adult (NSFW) content between adults is possible.
Pick a popular community SDXL checkpoint, or load any SDXL `.safetensors` from Hugging Face or Civitai, add LoRAs, batch, seed, and save to Drive.
An optional **Flux** section works on L4/A100.

**Runs on:** SDXL on a free **T4** (fp16, fp16-safe VAE). Flux on **L4/A100** (4-bit on ≤24 GB).

**Steps:** 1 → 2 → 3 (load model) → 4 (optional LoRAs) → 5 (txt2img) / 6 (img2img).

> ### ⚠️ Content policy (enforced)
> * **Sexual or suggestive content involving minors is prohibited.** Prompts, checkpoint names and LoRA names that contain minor/child-related terms (child, kid, teen, underage, loli, schoolgirl, ages under 18, …) are refused before generation. Rephrase if you hit a false positive.
> * **No sexual, nude or non-consensual imagery of real, identifiable people** (celebrities, acquaintances, anyone who has not consented), and no LoRAs trained on real people for that purpose.
> * You are responsible for complying with the model licenses and the law where you live."""),
    new_code_cell(INSTALL),
    new_code_cell(
        LOGIN
        + """
#@markdown For Civitai downloads add a `CIVITAI_TOKEN` secret too (Civitai > Account settings > API keys)."""
    ),
    new_code_cell(f"""#@title 3. Load a model
MODEL = "Juggernaut XL v9 (photoreal)"  #@param {json.dumps(IMAGE_MODELS)}
CUSTOM_SOURCE = ""  #@param {{type:"string"}}
#@markdown Custom: HF repo id (`owner/name`, diffusers format), HF file URL (`https://huggingface.co/.../blob/main/x.safetensors`), Civitai download URL (`https://civitai.com/api/download/models/<versionId>`) or a local `.safetensors` path.
CUSTOM_FAMILY = "sdxl"  #@param ["sdxl", "flux"]
SCHEDULER = "model default"  #@param {json.dumps(SCHEDULERS)}
DTYPE = "auto"  #@param ["auto", "fp16", "bf16", "fp32"]
OFFLOAD = "auto"  #@param ["auto", "none", "model", "sequential"]

from aurora.device import free_memory
from aurora.image import IMAGE_MODELS, ImageGenerator, set_scheduler

_src = CUSTOM_SOURCE.strip() if MODEL.startswith("Custom") else MODEL
if not _src:
    raise ValueError("Set CUSTOM_SOURCE or pick a model from the list.")
_key = (_src, CUSTOM_FAMILY, DTYPE, OFFLOAD)
if globals().get("IMG") is None or globals().get("_IMG_KEY") != _key:
    if globals().get("IMG") is not None:
        IMG.unload()
        IMG = None
        free_memory()
    IMG = ImageGenerator(_src, family=CUSTOM_FAMILY, dtype=DTYPE, offload=OFFLOAD,
                         scheduler=None if SCHEDULER == "model default" else SCHEDULER)
    _IMG_KEY = _key
elif SCHEDULER != "model default" and IMG.family == "sdxl":
    set_scheduler(IMG.pipe, SCHEDULER)
    IMG._img2img = None
spec = IMAGE_MODELS.get(_src)
print(f"Ready: {{_src}} ({{IMG.family}}, {{IMG.dtype}}, offload={{IMG.offload}})" + (f" | {{spec.notes}} | default {{spec.width}}x{{spec.height}}, {{spec.steps}} steps, CFG {{spec.guidance}}" if spec else ""))"""),
    new_code_cell("""#@title 4. (Optional) LoRAs
#@markdown HF repo id, HF file URL, Civitai download URL or local path. Leave blank to skip. Re-running replaces previously loaded LoRAs.
LORA_1 = ""  #@param {type:"string"}
LORA_1_WEIGHT = 0.8  #@param {type:"slider", min:-1.0, max:2.0, step:0.05}
LORA_1_FILE = ""  #@param {type:"string"}
#@markdown `LORA_x_FILE`: file name inside an HF repo that has several `.safetensors` (optional).
LORA_2 = ""  #@param {type:"string"}
LORA_2_WEIGHT = 0.8  #@param {type:"slider", min:-1.0, max:2.0, step:0.05}
LORA_2_FILE = ""  #@param {type:"string"}

IMG.clear_loras()
for src, wt, fn in ((LORA_1, LORA_1_WEIGHT, LORA_1_FILE), (LORA_2, LORA_2_WEIGHT, LORA_2_FILE)):
    if src.strip():
        IMG.add_lora(src.strip(), weight=wt, weight_name=fn.strip() or None)
IMG._img2img = None
print("Active LoRAs:", IMG.loras or "none")"""),
    new_code_cell("""#@title 5. Text → image
PROMPT = "photo of an elegant woman in a red evening dress on a rooftop at night, city lights bokeh, 85mm, film grain"  #@param {type:"string"}
NEGATIVE_PROMPT = "default"  #@param {type:"string"}
#@markdown `default` = generic quality negatives. Not used by Flux.
ADULT_SAFETY_NEGATIVES = True  #@param {type:"boolean"}
#@markdown Appends `child, teen, underage, …` to the negative prompt (recommended for NSFW work).
WIDTH = 0  #@param {type:"integer"}
HEIGHT = 0  #@param {type:"integer"}
#@markdown `0` = model default (SDXL likes ~1 MP: 1024x1024, 832x1216, 1216x832, 896x1152).
STEPS = 0  #@param {type:"integer"}
GUIDANCE = -1  #@param {type:"number"}
#@markdown `0` steps / `-1` guidance = model default.
SEED = -1  #@param {type:"integer"}
BATCH = 2  #@param {type:"slider", min:1, max:8, step:1}
#@markdown T4: keep batch ≤ 4 at 1024². Seeds are SEED, SEED+1, …
CLIP_SKIP = 0  #@param {type:"integer"}
#@markdown A1111-style clip skip (2 for many anime models); 0 = model default.
SAVE_TO_DRIVE = False  #@param {type:"boolean"}

import time

from aurora.image import DEFAULT_SDXL_NEGATIVE
from aurora.outputs import copy_to_drive, save_image, show_images
from aurora.safety import ADULT_NEGATIVE_PROMPT

neg = DEFAULT_SDXL_NEGATIVE if NEGATIVE_PROMPT.strip() == "default" else NEGATIVE_PROMPT
if ADULT_SAFETY_NEGATIVES:
    neg = f"{neg}, {ADULT_NEGATIVE_PROMPT}" if neg else ADULT_NEGATIVE_PROMPT

t0 = time.time()
images, metas = IMG.generate(PROMPT, negative_prompt=neg, width=WIDTH or None, height=HEIGHT or None,
                             steps=STEPS or None, guidance=None if GUIDANCE < 0 else GUIDANCE, seed=SEED,
                             batch_size=BATCH, clip_skip=CLIP_SKIP or None)
paths = [save_image(im, PROMPT, seed=m["seed"], metadata=m) for im, m in zip(images, metas)]
print(f"{len(images)} image(s) in {time.time() - t0:.0f}s | seeds {[m['seed'] for m in metas]}")
for p in paths:
    print("Saved:", p)
    if SAVE_TO_DRIVE:
        copy_to_drive(p, "Aurora/images")
show_images(images)"""),
    new_code_cell("""#@title 6. Image → image
INIT_IMAGE = ""  #@param {type:"string"}
#@markdown URL or `/content/...` path. Blank = upload a file. Type `last` to reuse the first image from step 5.
PROMPT = "same woman, oil painting style, dramatic lighting"  #@param {type:"string"}
NEGATIVE_PROMPT = "default"  #@param {type:"string"}
ADULT_SAFETY_NEGATIVES = True  #@param {type:"boolean"}
STRENGTH = 0.55  #@param {type:"slider", min:0.05, max:1.0, step:0.05}
#@markdown Lower = closer to the input image.
WIDTH = 0  #@param {type:"integer"}
HEIGHT = 0  #@param {type:"integer"}
STEPS = 0  #@param {type:"integer"}
GUIDANCE = -1  #@param {type:"number"}
SEED = -1  #@param {type:"integer"}
BATCH = 1  #@param {type:"slider", min:1, max:8, step:1}
SAVE_TO_DRIVE = False  #@param {type:"boolean"}

from PIL import Image

from aurora.image import DEFAULT_SDXL_NEGATIVE
from aurora.outputs import copy_to_drive, save_image, show_images
from aurora.safety import ADULT_NEGATIVE_PROMPT

if INIT_IMAGE.strip() == "last":
    init = images[0]
elif INIT_IMAGE.strip():
    init = INIT_IMAGE.strip()
else:
    from google.colab import files
    up = files.upload()
    init = Image.open(next(iter(up)))
if isinstance(init, Image.Image) and not (WIDTH and HEIGHT):
    # keep the input's aspect ratio at ~1 MP
    r = (1024 * 1024 / (init.width * init.height)) ** 0.5
    WIDTH, HEIGHT = int(init.width * r) // 64 * 64, int(init.height * r) // 64 * 64

neg = DEFAULT_SDXL_NEGATIVE if NEGATIVE_PROMPT.strip() == "default" else NEGATIVE_PROMPT
if ADULT_SAFETY_NEGATIVES:
    neg = f"{neg}, {ADULT_NEGATIVE_PROMPT}" if neg else ADULT_NEGATIVE_PROMPT
out, metas = IMG.generate(PROMPT, negative_prompt=neg, init_image=init, strength=STRENGTH,
                          width=WIDTH or None, height=HEIGHT or None, steps=STEPS or None,
                          guidance=None if GUIDANCE < 0 else GUIDANCE, seed=SEED, batch_size=BATCH)
for im, m in zip(out, metas):
    p = save_image(im, PROMPT, seed=m["seed"], metadata=m)
    print("Saved:", p)
    if SAVE_TO_DRIVE:
        copy_to_drive(p, "Aurora/images")
show_images(out)"""),
    new_markdown_cell("""## Optional: Flux

Pick **FLUX.1 schnell** or **FLUX.1 dev** in step 3 (or `CUSTOM_FAMILY = "flux"` with a Flux transformer `.safetensors`).

* Needs an **L4 or A100** runtime (bf16). On ≤ 24 GB GPUs Aurora loads the transformer and T5 encoder in 4-bit NF4 automatically.
* Both repos are **gated**: accept the license on Hugging Face and add `HF_TOKEN` (step 2). FLUX.1-dev is non-commercial.
* schnell: 4 steps, guidance 0. dev: ~28 steps, guidance 3-4. Negative prompts are ignored.
* Flux LoRAs load the same way as SDXL LoRAs in step 4 (they must be Flux LoRAs).

## Troubleshooting

* **CUDA out of memory**: lower `BATCH`, use ~1 MP sizes, set `OFFLOAD = "model"` (or `"sequential"`) and re-run step 3. *Runtime > Restart session* frees fragmented VRAM.
* **Black images on T4**: fp16 overflow. Aurora already swaps in the fp16-safe SDXL VAE; if a custom checkpoint still produces black images set `DTYPE = "fp32"` + `OFFLOAD = "model"`.
* **Civitai 401/403**: add a `CIVITAI_TOKEN` Colab secret; use the *download* URL (`/api/download/models/<versionId>`), not the page URL.
* **Prompt refused**: the minor-safety filter matched a term (it is deliberately strict, e.g. `teen`, `kid`, `schoolgirl`, ages < 18). Rephrase; there is no override.
* **Long prompts**: SDXL's CLIP encoders read the first 77 tokens; put the important words first.
"""),
]


def build(cells, name: str) -> None:
    nb = new_notebook(cells=cells)
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"name": name, "provenance": [], "gpuType": "T4", "private_outputs": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbformat.validate(nb)
    out = ROOT / "notebooks" / name
    out.parent.mkdir(exist_ok=True)
    nbformat.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    build(video_cells, "aurora_video_wan.ipynb")
    build(image_cells, "aurora_image.ipynb")
