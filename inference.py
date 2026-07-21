"""Inference entry point for the minimal Qwen2.5-VL reimplementation.

Loads the model built in ``model.py``, preprocesses a single image + prompt, and
runs greedy generation. The image path and prompt can be passed on the command
line; both default to the values used by the original single-file script.

Example:
    python inference.py --image_path example.png --prompt "Describe this image."
"""
import argparse

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import AutoTokenizer

from model import (
    MODEL_PATH,
    V,
    IMG_MEAN,
    IMG_STD,
    IMAGE_TOKEN_ID,
    smart_resize,
    MiniQwen25VL,
    load_weights,
)

DEFAULT_IMAGE_PATH = "example.png"
DEFAULT_PROMPT = "Describe this image."


def preprocess(img, tokenizer, prompt=DEFAULT_PROMPT):
    w, h = img.size  # PIL: (W, H)
    H, W = smart_resize(h, w)
    grid_h, grid_w = H // V.patch_size, W // V.patch_size
    x = TF.pil_to_tensor(img.resize((W, H), Image.BICUBIC)).float() / 255.0  # (3, H, W)
    x = TF.normalize(x, IMG_MEAN, IMG_STD)
    # patchify in merge-block order, temporal dim duplicated (matches the HF processor)
    x = x.view(3, grid_h // 2, 2, 14, grid_w // 2, 2, 14).permute(1, 4, 2, 5, 0, 3, 6)
    x = x.reshape(grid_h * grid_w, 3, 14, 14).unsqueeze(2).expand(-1, -1, 2, -1, -1)
    pixel_values = x.reshape(grid_h * grid_w, 3 * 2 * 14 * 14)
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt").input_ids[0].tolist()
    n_img = (grid_h // 2) * (grid_w // 2)
    new = []
    for t in ids:
        new.extend([IMAGE_TOKEN_ID] * n_img if t == IMAGE_TOKEN_ID else [t])
    input_ids = torch.tensor([new], dtype=torch.long)
    mm_type = (input_ids == IMAGE_TOKEN_ID).int()
    return input_ids, mm_type, grid_thw, pixel_values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal from-scratch Qwen2.5-VL-3B-Instruct inference.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the input image (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt to ask about the image (default: %(default)s).",
    )
    parser.add_argument(
        "--max_new",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    img = Image.open(args.image_path).convert("RGB")
    input_ids, mm_type, grid_thw, pixel_values = preprocess(img, tok, args.prompt)

    model = MiniQwen25VL()
    load_weights(model)
    model = model.to("cuda", dtype=torch.bfloat16).eval()

    out = model.generate(
        input_ids.to("cuda"),
        mm_type.to("cuda"),
        grid_thw.to("cuda"),
        pixel_values.to("cuda"),
        max_new=args.max_new,
    )
    print([tok.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)])


if __name__ == "__main__":
    main()
