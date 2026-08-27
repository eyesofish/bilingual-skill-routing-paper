#!/usr/bin/env python3
"""Build a sanitized skill metadata catalog from explicitly allowed sources."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FRONTMATTER_DELIMITER = "---"
LIST_FIELDS = {"aliases", "tags"}


def _parse_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_inline_list(value: str) -> list[str]:
    inner = value.strip()[1:-1].strip()
    if not inner:
        return []
    return [_parse_scalar(item.strip()) for item in inner.split(",") if item.strip()]


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by the benchmark's SKILL.md files."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
        return {}

    try:
        end = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == FRONTMATTER_DELIMITER
        )
    except StopIteration:
        return {}

    metadata: dict[str, Any] = {}
    index = 1
    while index < end:
        line = lines[index]
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):(?:\s*(.*))?$", line)
        if not match:
            index += 1
            continue

        key, raw_value = match.group(1), (match.group(2) or "").strip()
        if raw_value in {">", "|", ">-", "|-"}:
            block: list[str] = []
            index += 1
            while index < end and (
                not lines[index].strip() or lines[index].startswith((" ", "\t"))
            ):
                block.append(lines[index].strip())
                index += 1
            metadata[key] = " ".join(part for part in block if part).strip()
            continue

        if not raw_value and key in LIST_FIELDS:
            values: list[str] = []
            index += 1
            while index < end:
                item_match = re.match(r"^\s*-\s*(.+?)\s*$", lines[index])
                if not item_match:
                    break
                values.append(_parse_scalar(item_match.group(1)))
                index += 1
            metadata[key] = values
            continue

        if raw_value.startswith("[") and raw_value.endswith("]"):
            metadata[key] = _parse_inline_list(raw_value)
        else:
            metadata[key] = _parse_scalar(raw_value)
        index += 1

    return metadata


def _skill_directories(root: Path, allow: list[str]) -> list[Path]:
    if allow == ["*"]:
        return sorted(
            child for child in root.iterdir() if child.is_dir() and (child / "SKILL.md").is_file()
        )
    return [root / name for name in allow]


def build_catalog(config_path: Path) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    catalog: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for source in config["sources"]:
        label = source["label"]
        root = Path(source["root"]).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"source root does not exist: {root}")

        for skill_dir in _skill_directories(root, source["allow"]):
            skill_path = skill_dir / "SKILL.md"
            if not skill_path.is_file():
                raise FileNotFoundError(f"allowed skill is missing SKILL.md: {skill_path}")
            metadata = parse_frontmatter(skill_path.read_text(encoding="utf-8"))
            name = str(metadata.get("name") or skill_dir.name).strip()
            if not name:
                raise ValueError(f"skill has no name: {skill_path}")
            if name in seen_names:
                raise ValueError(f"duplicate skill name in benchmark catalog: {name}")
            seen_names.add(name)

            catalog.append(
                {
                    "name": name,
                    "description": str(metadata.get("description") or "").strip(),
                    "when_to_use": str(metadata.get("when_to_use") or "").strip(),
                    "aliases": list(metadata.get("aliases") or []),
                    "tags": list(metadata.get("tags") or []),
                    "source": label,
                    "source_path": f"{label}/{skill_dir.name}/SKILL.md",
                }
            )

    return sorted(catalog, key=lambda item: item["name"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/sources.json"))
    parser.add_argument("--output", type=Path, default=Path("data/catalog.json"))
    args = parser.parse_args()

    catalog = build_catalog(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"schema_version": 1, "skills": catalog}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(catalog)} skills to {args.output}")


if __name__ == "__main__":
    main()

