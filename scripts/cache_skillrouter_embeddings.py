#!/usr/bin/env python3
"""Precompute metadata embeddings for the SkillRouter hard-tier catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from evaluate_skillrouter_multilingual import (
    detect_device,
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
        default=Path("data/external/cache/multilingual-e5-small-hard-metadata.npy"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    model_spec = load_model_specs(args.model_config)["multilingual_e5_small"]

    skill_ids, documents, _ = load_pool(args.shards)
    formatted_documents = [f"passage: {document}" for document in documents]
    device = detect_device(args.device)
    model = SentenceTransformer(
        model_spec["model_id"],
        revision=model_spec["revision"],
        device=device,
    )
    model.max_seq_length = 512
    embeddings = model.encode(
        formatted_documents,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)

    metadata = {
        "model": model_spec["model_id"],
        "model_revision": model_revision(model),
        "device": device,
        "batch_size": args.batch_size,
        "max_seq_length": model.max_seq_length,
        "skill_count": len(skill_ids),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "sha256": file_sha256(args.output),
        "skill_id_sha256": ordered_text_sha256(skill_ids),
        "document_sha256": ordered_text_sha256(formatted_documents),
        "document_fields": "passage: name. description",
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {metadata_path}")


if __name__ == "__main__":
    main()
