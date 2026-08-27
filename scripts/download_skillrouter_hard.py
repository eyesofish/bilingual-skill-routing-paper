#!/usr/bin/env python3
"""Download and verify the public SkillRouter hard-tier evaluation files."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums(path: Path) -> dict[str, str]:
    checksums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split(maxsplit=1)
            checksums[relative.strip()] = digest
    return checksums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("data/external/skillrouter")
    )
    parser.add_argument(
        "--checksums", type=Path, default=Path("data/skillrouter-hard.sha256")
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("config/dataset_revisions.json"),
    )
    args = parser.parse_args()

    dataset = json.loads(args.dataset_config.read_text(encoding="utf-8"))[
        "skillrouter_eval_core"
    ]
    base_url = (
        f"https://huggingface.co/datasets/{dataset['repo_id']}/resolve/"
        f"{dataset['revision']}"
    )
    checksums = load_checksums(args.checksums)
    for relative, expected in checksums.items():
        output = args.output / relative
        if not output.is_file() or sha256(output) != expected:
            print(f"downloading {relative}")
            temporary = output.with_suffix(output.suffix + ".tmp")
            output.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{base_url}/{relative}", temporary)
            temporary.replace(output)
        actual = sha256(output)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {relative}: {actual}")
    print(f"verified {len(checksums)} SkillRouter files in {args.output}")


if __name__ == "__main__":
    main()
