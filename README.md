# 🌌 Aurora

**Colab-first generative media.** Open a notebook, pick a GPU, generate.

* 🎬 **Video**: text-to-video and image-to-video with **[Wan 2.1 / 2.2](https://huggingface.co/Wan-AI)** via 🤗 diffusers.
* 🖼️ **Images**: SDXL text-to-image and image-to-image with popular community checkpoints, any HF/Civitai `.safetensors`, LoRAs, batches and seeds. Optional **Flux**. No safety checker, so adult content is supported, with a hard block on anything involving minors (see [Content policy](#content-policy)).

| Notebook | Open |
|---|---|
| Wan video generation (T2V + I2V) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/subash1327/aurora/blob/main/notebooks/aurora_video_wan.ipynb) |
| Image generation (SDXL / Flux, NSFW-capable) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/subash1327/aurora/blob/main/notebooks/aurora_image.ipynb) |

## Quickstart

1. Click a badge above.
2. **Runtime → Change runtime type → T4 GPU** (free) or **L4 / A100** (Colab Pro).
3. *(Optional)* Add a Hugging Face token: 🔑 **Secrets** sidebar → `HF_TOKEN` → enable notebook access. Required for gated models (FLUX). For Civitai downloads add `CIVITAI_TOKEN`.
4. Run the cells top to bottom: **install → login → load model → generate**. Re-run only the generate cell for new prompts.
5. Outputs go to `/content/aurora_outputs/{videos,images}` with a `.json` sidecar of every parameter (seed included). Tick `SAVE_TO_DRIVE` to copy them to `MyDrive/Aurora/`.

## Uploading files from your computer

Both notebooks accept files straight from your computer. You don't need to host them anywhere first.

| What | Where | How |
|---|---|---|
| Start image for **image-to-video** | Video notebook, step **3b** | file picker or drag & drop, then step 4 |
| Init image for **img2img** | Image notebook, step **6a** | file picker or drag & drop, then step 6 |
| **LoRA** / **SDXL checkpoint** `.safetensors` | Image notebook, step **2b** | then type `uploaded` (latest) or `uploaded:2` (2nd file) in `LORA_1`/`LORA_2` or `CUSTOM_SOURCE` |

* **file picker** uses Colab's native upload dialog (`google.colab.files.upload`). **drag & drop widget** shows an `ipywidgets.FileUpload` drop zone. Drop the file, then run the next cell.
* **Precedence:** an uploaded image wins over the `INPUT_IMAGE` / `INIT_IMAGE` field (URL, `/content/...` path, Drive path `/content/drive/MyDrive/...`, or `last`). Untick `USE_UPLOADED` to ignore uploads. If nothing is uploaded and the field is blank, a file picker opens.
* Uploads are saved under `/content/uploads/{images,loras,checkpoints}/`. Multi-GB checkpoints upload slowly through the browser, so put them on Google Drive and use the Drive path instead.
* Outside Colab the helpers fall back to the drag & drop widget (if `ipywidgets` is installed) or simply to paths/URLs. They never crash.

## GPU requirements

### Video (Wan)

| Model (HF repo) | Mode | Native res / fps | Free T4 (15 GB) | L4 (24 GB) | A100 40/80 GB |
|---|---|---|---|---|---|
| **Wan2.1-T2V-1.3B** (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) **default** | T2V | 480×832, 81 f @ 16 fps | ✅ fp16, 8-bit text encoder | ✅ | ✅ |
| **Wan2.2-TI2V-5B** (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`) default for I2V | T2V + I2V | 704×1280, 121 f @ 24 fps | ⚠️ 4-bit transformer, slow, use ≤49 frames | ✅ recommended | ✅ |
| Wan2.1-T2V-14B | T2V | 720×1280 | ❌ | ⚠️ 4-bit + offload | ✅ |
| Wan2.1-I2V-14B-480P / 720P | I2V | 480p / 720p | ❌ | ⚠️ 4-bit + offload | ✅ |
| Wan2.2-T2V-A14B / I2V-A14B (MoE, 2×14B) | T2V / I2V | 720p | ❌ | ❌ | ✅ 80 GB, 40 GB with offload |

`auto` settings pick the dtype (fp16 on T4, bf16 on L4/A100), bitsandbytes quantization and CPU offload from your GPU's VRAM. You can override every choice in the notebook.

### Images

| Model | Style | T4 | Notes |
|---|---|---|---|
| **Juggernaut XL v9** (`RunDiffusion/Juggernaut-XL-v9`) **default** | photoreal | ✅ | |
| RealVisXL V5.0 (`SG161222/RealVisXL_V5.0`) | photoreal | ✅ | |
| Animagine XL 4.0 (`cagliostrolab/animagine-xl-4.0`) | anime | ✅ | Danbooru-style tags |
| Illustrious XL v0.1 (`OnomaAIResearch/Illustrious-xl-early-release-v0`) | anime | ✅ | clip skip 2 |
| DreamShaper XL v2 Turbo (`Lykon/dreamshaper-xl-v2-turbo`) | general | ✅ | 6-8 steps, CFG 2 |
| SDXL Base 1.0 | general | ✅ | |
| Custom: any SDXL `.safetensors` (HF URL, Civitai download URL, local) | – | ✅ | loaded with `from_single_file` |
| FLUX.1 schnell / dev | general | ❌ | L4/A100, bf16, NF4 on ≤24 GB; gated (HF_TOKEN) |

SDXL runs in fp16 on a T4 using the fp16-safe VAE (`madebyollin/sdxl-vae-fp16-fix`) to avoid black images.

## Using the package directly

The notebooks are thin wrappers around the `aurora` package:

```bash
pip install -r https://raw.githubusercontent.com/subash1327/aurora/main/requirements.txt
pip install git+https://github.com/subash1327/aurora
```

```python
from aurora.video import WanGenerator
from aurora.outputs import save_video

gen = WanGenerator(model="Wan2.1-T2V-1.3B", mode="t2v")          # dtype/offload/quantization = auto
frames, params = gen.generate("a red fox running through fresh snow, cinematic", num_frames=49, seed=42)
save_video(frames, params["prompt"], fps=16, metadata=params)

from aurora.image import ImageGenerator
img = ImageGenerator("Juggernaut XL v9 (photoreal)")
img.add_lora("https://civitai.com/api/download/models/123456", weight=0.7)
images, metas = img.generate("portrait of a woman, golden hour, 85mm", batch_size=2, seed=7)

# local uploads (Colab file picker / drag & drop widget; plain paths work everywhere)
from aurora.inputs import resolve_image, upload
uploads = upload("image")                          # opens the picker in Colab
init = resolve_image("/path/or/url.png", uploader=uploads)   # uploaded file wins if present
images, metas = img.generate("oil painting style", init_image=init, strength=0.55)
```

Layout:

```
aurora/
  device.py     GPU/VRAM/RAM detection, dtype choice, HF login from Colab secrets
  memory.py     offload strategy + VAE tiling/slicing
  video.py      Wan model registry, quantization plan, WanGenerator (T2V / I2V)
  image.py      SDXL/Flux model registry, ImageGenerator (txt2img, img2img, LoRA, schedulers)
  downloads.py  HF / Civitai / URL downloads for checkpoints and LoRAs
  inputs.py     uploads from your computer (Colab picker / drag & drop), image + "uploaded" source resolution
  outputs.py    PNG/MP4 saving with metadata, inline display, Google Drive copy
  safety.py     prompt filter (minor-related terms are refused)
notebooks/      Colab notebooks (generated by tools/build_notebooks.py)
tests/          safety filter, planning and upload-helper tests
```

## Content policy

Aurora can generate adult content. These rules are not optional:

1. **Sexual or sexualised content involving minors is prohibited**, including fictional, drawn, anime or "aged-up" characters presented as minors. `aurora/safety.py` refuses any prompt, checkpoint name or LoRA name containing minor/child-related terms (child, kid, teen, underage, loli, shota, schoolgirl, explicit ages under 18, and more) before generation. The filter is deliberately strict; rephrase if you hit a false positive. Removing or bypassing it is a violation of this policy.
2. **No sexual, nude or non-consensual content of real, identifiable people**, such as celebrities, public figures, acquaintances, or anyone who has not explicitly consented. Do not use LoRAs or reference images of real people for that purpose.
3. No content intended to harass, defame, or deceive, such as fake "evidence" or impersonation.
4. Follow each model's license (FLUX.1-dev is non-commercial; check Civitai model licenses) and the law where you live.

Users are solely responsible for what they generate.

## Troubleshooting

**CUDA out of memory**
* Video: lower `NUM_FRAMES` (49 or 33) and/or resolution; set `OFFLOAD = "sequential"`; set `QUANTIZE = "4bit"`; re-run the load cell.
* Images: lower `BATCH`, stay around 1 MP (1024², 832×1216), `OFFLOAD = "model"`.
* *Runtime → Restart session* clears fragmented VRAM after an OOM.

**Session crashes while loading (system RAM)**: free Colab has about 12.7 GB of RAM. Wan's UMT5 text encoder alone is about 11 GB at 16-bit, so keep `QUANTIZE = "auto"` (8-bit text encoder) on T4, or use a High-RAM / L4 runtime.

**Black / NaN output on T4**: T4 has no fast bf16, so Aurora uses fp16. Some checkpoints overflow in fp16. Try `DTYPE = "fp32"` with offload, or an L4/A100.

**Slow**: Wan 1.3B takes several minutes per 5 s clip on a T4 (more with 5B in 4-bit). Use 20–25 steps for drafts.

**Gated model / 401**: accept the license on the model's HF page and add `HF_TOKEN`. For Civitai, add `CIVITAI_TOKEN` and use the `/api/download/models/<versionId>` link.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
python tools/build_notebooks.py   # regenerate notebooks after editing their cell sources
```

## License

[MIT](LICENSE) © 2026 Subash. Model weights are under their own licenses.
