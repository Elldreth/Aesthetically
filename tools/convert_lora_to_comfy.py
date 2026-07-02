"""Convert an Artifex-trained LoRA (diffusers-style keys) to the kohya dialect
that ComfyUI / A1111 expect.

Artifex saves UNet LoRA keys like:
    unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight
ComfyUI wants:
    lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight
plus a per-module `.alpha`. This remaps the keys and writes a new *_comfy.safetensors;
the original is untouched.

    python -m tools.convert_lora_to_comfy --in  D:\\repos\\Artifex\\models\\lora\\my.safetensors
                                          --out D:\\Comfy\\models\\loras\\my.safetensors
                                          [--alpha 32]   # default: rank/2 (Artifex's convention)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def convert(state: dict, alpha: float | None) -> dict:
    out: dict = {}
    seen_alpha: set[str] = set()
    skipped = []
    for k, v in state.items():
        if k.endswith(".lora.down.weight"):
            module, suffix = k[:-len(".lora.down.weight")], "lora_down.weight"
        elif k.endswith(".lora.up.weight"):
            module, suffix = k[:-len(".lora.up.weight")], "lora_up.weight"
        else:
            skipped.append(k)
            continue
        if module.startswith("unet."):
            module = module[len("unet."):]
            prefix = "lora_unet_"
        elif module.startswith("text_encoder."):
            module = module[len("text_encoder."):]
            prefix = "lora_te_"
        else:
            prefix = "lora_unet_"
        base = prefix + module.replace(".", "_")
        out[base + "." + suffix] = v
        if base not in seen_alpha:
            seen_alpha.add(base)
            rank = v.shape[0] if suffix == "lora_down.weight" else v.shape[1]
            a = alpha if alpha is not None else rank / 2          # Artifex uses alpha=rank/2
            out[base + ".alpha"] = torch.tensor(float(a), dtype=v.dtype)
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst")
    ap.add_argument("--alpha", type=float, default=None,
                    help="network alpha (default rank/2 = Artifex's training scale)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src.with_name(src.stem + "_comfy.safetensors")
    state = load_file(str(src))
    converted, skipped = convert(state, args.alpha)
    n_alpha = sum(1 for k in converted if k.endswith(".alpha"))
    save_file(converted, str(dst), metadata={"format": "pt"})
    print(f"in : {len(state)} tensors")
    print(f"out: {len(converted)} tensors ({n_alpha} alpha, {len(converted) - n_alpha} weights)")
    if skipped:
        print(f"skipped {len(skipped)} unrecognized keys (e.g. {skipped[0]})")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
