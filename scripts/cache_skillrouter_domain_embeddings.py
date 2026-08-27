#!/usr/bin/env python3
"""Cache SkillRouter-Embedding-0.6B vectors for name and description metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from evaluate_skillrouter_multilingual import (
    detect_device,
    encode_skillrouter_texts,
    load_pool,
    load_model_specs,
    model_revision,
    ordered_text_sha256,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards", type=Path, default=Path("data/external/skillrouter/hard")
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/model_revisions.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/cache/skillrouter-0.6b-hard-metadata.npy"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    model_spec = load_model_specs(args.model_config)["skillrouter_embedding_0_6b"]

    skill_ids, _, metadata = load_pool(args.shards)
    documents = [
        f"{item['name']} | {item['description']}".strip() for item in metadata
    ]
    device = detect_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        trust_remote_code=True,
        padding_side="left",
    )
    model = AutoModel.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model = model.eval().to(device)
    embeddings = encode_skillrouter_texts(
        documents, tokenizer, model, args.batch_size, args.max_length
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    metadata = {
        "model": model_spec["model_id"],
        "model_revision": model_revision(model),
        "device": device,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "skill_count": len(documents),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "sha256": file_sha256(args.output),
        "skill_id_sha256": ordered_text_sha256(skill_ids),
        "document_sha256": ordered_text_sha256(documents),
        "document_fields": "name | description",
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {metadata_path}")


if __name__ == "__main__":
    main()
