#!/usr/bin/env python3
"""Download and verify the public SkillRouter hard-tier evaluation files."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


BASE_URL = "https://huggingface.co/datasets/pipizhao/SkillRouter-Eval-Core/resolve/main"


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


def download(relative: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    urllib.request.urlretrieve(f"{BASE_URL}/{relative}", temporary)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("data/external/skillrouter")
    )
    parser.add_argument(
        "--checksums", type=Path, default=Path("data/skillrouter-hard.sha256")
    )
    args = parser.parse_args()

    checksums = load_checksums(args.checksums)
    for relative, expected in checksums.items():
        output = args.output / relative
        if not output.is_file() or sha256(output) != expected:
            print(f"downloading {relative}")
            download(relative, output)
        actual = sha256(output)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {relative}: {actual}")
    print(f"verified {len(checksums)} SkillRouter files in {args.output}")


if __name__ == "__main__":
    main()

