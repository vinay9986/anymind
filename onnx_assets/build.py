from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def _load_config() -> dict:
    cfg_path = Path(os.environ.get("ONNX_ASSETS_CONFIG", "onnx_assets/config.json"))
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("onnx_assets/config.json must be a JSON object")
    return data


class _Wrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return out.last_hidden_state


def main() -> None:
    cfg = _load_config()

    model_id = str(cfg.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("config.json must set model_id")

    output_dir = Path(str(cfg.get("output_dir") or "onnx_assets_out"))
    output_dir.mkdir(parents=True, exist_ok=True)

    opset = int(cfg.get("opset") or 17)
    max_length = int(cfg.get("max_length") or 256)

    print(f"[onnx_assets] downloading model_id={model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    print(f"[onnx_assets] saving tokenizer to {output_dir}")
    tokenizer.save_pretrained(str(output_dir))

    sample = tokenizer(
        ["hello world"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    wrapper = _Wrapper(model)

    onnx_path = output_dir / "model.onnx"

    input_names = ["input_ids", "attention_mask"]
    example_args = (sample["input_ids"], sample["attention_mask"])
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    }

    if "token_type_ids" in sample:
        input_names.append("token_type_ids")
        example_args = (
            sample["input_ids"],
            sample["attention_mask"],
            sample["token_type_ids"],
        )
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "seq"}

    print(f"[onnx_assets] exporting ONNX to {onnx_path} (opset={opset})")
    torch.onnx.export(
        wrapper,
        example_args,
        str(onnx_path),
        input_names=input_names,
        output_names=["last_hidden_state"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )

    if not (output_dir / "tokenizer.json").exists():
        raise RuntimeError(
            "Expected tokenizer.json to be created by tokenizer.save_pretrained()"
        )

    print("[onnx_assets] done")


if __name__ == "__main__":
    main()
