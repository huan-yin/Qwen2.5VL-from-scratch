# Qwen2.5-VL from scratch

A minimal, from-scratch reimplementation of **Qwen2.5-VL-3B-Instruct** using only
`torch.nn.Module` — no `Qwen2_5_VLForConditionalGeneration`, no
`Qwen2_5_VLProcessor`. Pretrained weights are loaded straight from the
safetensors checkpoint, so the model produces real outputs rather than being a
random toy.

The goal is **readability**: the entire multimodal architecture (vision encoder,
language model, vision-language connector, M-RoPE, KV cache, generation loop)
fits in a single [`model.py`](model.py) file that mirrors the original design
without the abstraction layers of a framework.

## Features

- **Pure `nn.Module` implementation** — every layer is written out by hand:
  `RMSNorm`, GQA attention, SwiGLU MLP, 3D Conv patch embedding, rotary
  embeddings, and the spatial-merge patch merger.
- **Vision encoder** with windowed attention (window size 112) plus full
  attention on layers `(7, 15, 23, 31)`, 2D rotary position embeddings, and a
  2×2 spatial merge that cuts the number of visual tokens by 4×.
- **Language model** with grouped-query attention (16 query heads / 2 KV heads),
  **M-RoPE** (multidimensional rotary embedding with a temporal/height/width
  channel split of `(16, 24, 24)`), and a per-layer KV cache for fast
  autoregressive decoding.
- **Weight loading** directly from the HuggingFace safetensors checkpoint, with
  tied input/output embeddings.
- **Generation loop** with greedy decoding and a simple repetition penalty.
- **CLI inference** over a single image + prompt.

## Architecture

The constants below are hardcoded in [`model.py`](model.py) and match
`Qwen/Qwen2.5-VL-3B-Instruct`'s `config.json`.

| Component       | Setting                                                        |
| --------------- | ------------------------------------------------------------- |
| **Vision**      |                                                               |
| depth           | 32                                                            |
| hidden size     | 1280                                                          |
| num heads       | 16 (head dim 80)                                              |
| intermediate    | 3420                                                          |
| patch size      | 14 × 14, temporal patch 2                                     |
| spatial merge   | 2 × 2                                                         |
| window size     | 112                                                           |
| full-attn layers| `(7, 15, 23, 31)`                                             |
| output hidden   | 2048                                                          |
| **Language**    |                                                               |
| num layers      | 36                                                            |
| hidden size     | 2048                                                          |
| num heads       | 16 query / 2 KV (grouped-query)                               |
| head dim        | 128                                                           |
| intermediate    | 11008                                                         |
| vocab size      | 151936                                                        |
| RoPE theta      | 1,000,000                                                     |
| M-RoPE split    | `(16, 24, 24)` — temporal, height, width                     |

## Project structure

```
.
├── model.py          # Architecture, preprocessing helpers, weight loading
├── inference.py      # CLI entry point: preprocess image + prompt, run generation
├── requirements.txt  # Python dependencies
├── example.png       # Sample image used by the default prompt
└── LICENSE           # Apache 2.0
```

### `model.py`

Holds the architecture config (`class V` for vision, `class T` for text),
preprocessing helpers (`smart_resize`, `vision_position_ids`, `cu_seqlens`,
`window_index`, `rope_index`), the vision encoder (`VisionTransformer` and
submodules), the language model (`TextModel` and submodules), the full
`MiniQwen25VL` module with its `generate` method, and `load_weights`.

### `inference.py`

Preprocesses a single image + text prompt, builds the chat template, expands the
`<|image_pad|>` token to the right number of visual tokens, runs greedy
generation, and prints the decoded answer.


## Usage

Run inference with the bundled example image and default prompt:

```bash
python inference.py
```

Or pass your own image and prompt:

```bash
python inference.py \
    --image_path path/to/image.jpg \
    --prompt "Describe this image." \
    --max_new 128
```

Arguments:

| Flag           | Default             | Description                                  |
| -------------- | ------------------- | -------------------------------------------- |
| `--image_path` | `example.png`       | Path to the input image                      |
| `--prompt`     | `Describe this image.` | Text prompt to ask about the image        |
| `--max_new`    | `128`               | Maximum number of new tokens to generate     |

The decoded answer is printed to stdout.

## How it works

1. **Preprocess** ([`inference.py`](inference.py#L32)): the image is
   `smart_resize`d to keep its aspect ratio with both sides divisible by 28,
   normalized with the CLIP mean/std, then patchified in merge-block order so
   each row is one 2×2×14×14×temporal patch — matching the HF processor exactly.
2. **Encode** ([`VisionTransformer`](model.py#L234)): patches are embedded, run
   through 32 transformer blocks (windowed attention except on full-attention
   layers), then merged 4-to-1 by the `Merger` into the language model's hidden
   size (2048).
3. **Inject**: the `<|image_pad|>` placeholder tokens in the input sequence are
   replaced by the visual embeddings via `masked_scatter`.
4. **Position** ([`rope_index`](model.py#L385)): text tokens get 1D positions
   replicated across the three M-RoPE axes; image tokens get 3D
   temporal/height/width positions.
5. **Generate** ([`MiniQwen25VL.generate`](model.py#L358)): the prompt is run
   through the LLM once with a KV cache, then tokens are decoded one at a time
   with greedy argmax and a repetition penalty, stopping at the EOS token.

## Notes & limitations

- Inference is **greedy** (with repetition penalty) — no beam search, sampling,
  or top-k/top-p. Best for deterministic single answers.
- Only **single-image, single-turn** inference is implemented (batch size 1).
- The model runs in `bfloat16` on CUDA. CPU/MPS are not tested.
- Designed for **learning and inspection**, not throughput — there are fused
  kernels and FlashAttention paths in the upstream implementation that this
  intentionally does not use.
