"""Write what the Lambda image serves from a Qwen3.5 strands-decider checkpoint.

    python prepare_model.py MODEL REVISION OUT_DIR [VISION] [DECODER_PRECISION]

MODEL is a Hugging Face Hub id. The script writes OUT_DIR/torso, the torso with its LoRA folded
in, as the Qwen3_5ForCausalLM checkpoint llama.cpp's convert_hf_to_gguf.py reads, and
OUT_DIR/assets, what HiddenStateEngine.from_assets reads. With VISION `true`, OUT_DIR/vision
links to the base model at the revision the checkpoint records: its vision tower, which the
adapter leaves as it is, for convert_hf_to_gguf.py --mmproj. The model is named for
DECODER_PRECISION, the precision the Dockerfile gives the decoder. The Dockerfile runs it.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import transformers
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import load_file

from strands_decider.hidden_engine import save_assets

# The checkpoint's pointer head normalizes with torch's LayerNorm, whose eps it does not save.
HEAD_EPS = 1e-5


def base_revision(checkpoint: str, config: dict) -> str | None:
    """The base model revision the adapter was trained on: the config's, else the one the
    checkpoint's provenance.json records for the same base model."""
    if config.get("base_revision"):
        return config["base_revision"]
    path = os.path.join(checkpoint, "provenance.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        provenance = json.load(f)
    if provenance.get("base_model") != config["base_model"]:
        return None
    return provenance.get("base_model_revision")


def main(
    model_id: str, revision: str, out: str, vision: str = "false", decoder: str = "q8_0"
) -> None:
    if vision not in ("true", "false"):
        raise SystemExit(f"VISION must be true or false, not {vision!r}")
    checkpoint = snapshot_download(model_id, revision=revision)
    with open(os.path.join(checkpoint, "strands_decider_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    if config.get("head_type", "pointer") != "pointer":
        raise SystemExit(f"{model_id}: head_type {config['head_type']!r}; only pointer heads load")
    base, base_rev = config["base_model"], base_revision(checkpoint, config)
    base_config = transformers.AutoConfig.from_pretrained(base, revision=base_rev)
    if base_config.model_type not in ("qwen3_5", "qwen3_5_text"):
        raise SystemExit(f"{model_id}: a {base_config.model_type} torso; only Qwen3.5 is supported")

    # The base checkpoint is multimodal; its causal-LM class maps the text tower's weights.
    lm = transformers.Qwen3_5ForCausalLM.from_pretrained(
        base,
        config=base_config.get_text_config(),
        revision=base_rev,
        dtype=getattr(torch, config["torch_dtype"]),
    )
    if config.get("use_lora", True):
        adapter = PeftModel.from_pretrained(lm.model, os.path.join(checkpoint, "lora"))
        lm.model = adapter.merge_and_unload()
    # llama.cpp's embeddings are the decoder's final normed hidden states, which the head reads.
    lm.to(torch.float16).save_pretrained(os.path.join(out, "torso"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(os.path.join(out, "torso"))

    head = load_file(os.path.join(checkpoint, "head.safetensors"))
    save_assets(
        os.path.join(out, "assets"),
        tokenizer.backend_tokenizer,
        {k: v.float().numpy() for k, v in head.items()},
        model_name=f"{model_id.split('/')[-1]}-{decoder}",
        pointer_dim=config["pointer_dim"],
        eps=HEAD_EPS,
        max_length=config["max_length"],
        temperature=config["temperature"],
        temperature_by_kind=config.get("temperature_by_kind", {}),
        ordinal_smoothing=config.get("ordinal_smoothing", 0.0),
    )

    if vision == "true":
        # The weights, the config, and the image processor's preprocessor_config.json.
        tower = snapshot_download(
            base, revision=base_rev, allow_patterns=["*.json", "*.safetensors", "*.txt"]
        )
        os.symlink(tower, os.path.join(out, "vision"))


if __name__ == "__main__":
    main(*sys.argv[1:])
