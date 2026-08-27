#!/usr/bin/env python3
"""Audit the anonymous AAAI paper and reproducibility package."""

from __future__ import annotations

import argparse
import re
import subprocess
import zipfile
from pathlib import Path


FORBIDDEN_SOURCE_PATTERNS = {
    "forbidden package": r"\\usepackage(?:\[[^\]]*\])?\{(?:hyperref|geometry|fullpage|balance|CJK|ulem|wrapfig)\}",
    "manual spacing": r"\\(?:addtolength|baselinestretch|columnsep|newpage|clearpage|pagebreak)\b",
    "negative vertical spacing": r"\\v(?:space|skip)\s*\{\s*-",
    "identity": r"@microsoft\.com|/Users/[^/\s]+",
}
PDF_IDENTITY_PATTERNS = [
    r"@microsoft\.com",
    r"/Users/[^/\s]+",
]


def identity_patterns() -> list[str]:
    patterns = list(PDF_IDENTITY_PATTERNS)
    local = Path(".anonymous-identities")
    if local.is_file():
        patterns.extend(
            line.strip()
            for line in local.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return patterns


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def parse_pdfinfo(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument(
        "--archive", type=Path, default=Path("build/aaai-submission.zip")
    )
    args = parser.parse_args()

    source = "\n".join(
        (args.paper_dir / name).read_text(encoding="utf-8")
        for name in ("main.tex", "supplement.tex", "reproducibility-checklist.tex")
    )
    for label, pattern in FORBIDDEN_SOURCE_PATTERNS.items():
        if re.search(pattern, source, flags=re.IGNORECASE):
            raise ValueError(f"{label} found in paper/main.tex: {pattern}")

    log = (args.paper_dir / "main.log").read_text(encoding="utf-8")
    bad_log_patterns = [
        r"LaTeX Warning: Citation .* undefined",
        r"LaTeX Warning: Reference .* undefined",
        r"There were undefined references",
        r"Rerun to get citations correct",
        r"Overfull \\hbox",
    ]
    for pattern in bad_log_patterns:
        if re.search(pattern, log):
            raise ValueError(f"paper log contains: {pattern}")

    aux = (args.paper_dir / "main.aux").read_text(encoding="utf-8")
    page_match = re.search(
        r"\\newlabel\{lastcontentpage\}\{\{[^}]*\}\{(\d+)\}", aux
    )
    if not page_match:
        raise ValueError("could not find lastcontentpage label in main.aux")
    content_page = int(page_match.group(1))
    if content_page > 7:
        raise ValueError(f"main content ends on page {content_page}, above page 7")

    pdf = args.paper_dir / "main.pdf"
    info = parse_pdfinfo(run("pdfinfo", str(pdf)))
    pages = int(info["Pages"])
    if pages > 9:
        raise ValueError(f"paper has {pages} pages, above the 9-page limit")
    if "letter" not in info["Page size"].lower():
        raise ValueError(f"paper is not US Letter: {info['Page size']}")
    if info.get("Encrypted", "").lower() != "no":
        raise ValueError("paper PDF is encrypted")

    fonts = run("pdffonts", str(pdf)).splitlines()
    for line in fonts[2:]:
        columns = line.split()
        if not columns:
            continue
        if "Type 3" in line or "Type3" in line:
            raise ValueError(f"Type 3 font found: {line}")
        if len(columns) < 5 or columns[-5].lower() != "yes":
            raise ValueError(f"unembedded font found: {line}")

    extracted = run("pdftotext", str(pdf), "-")
    patterns = identity_patterns()
    for pattern in patterns:
        if re.search(pattern, extracted, flags=re.IGNORECASE):
            raise ValueError(f"identity pattern found in PDF: {pattern}")
    if not re.search(r"anonymous submission", extracted, flags=re.IGNORECASE):
        raise ValueError("anonymous author marker missing from PDF")

    checklist = (args.paper_dir / "reproducibility-checklist.tex").read_text(
        encoding="utf-8"
    )
    if checklist.count("Type your response here") != 3:
        raise ValueError("unanswered reproducibility checklist items remain")
    checklist_pdf = args.paper_dir / "reproducibility-checklist.pdf"
    if not checklist_pdf.is_file():
        raise FileNotFoundError(checklist_pdf)

    archives = [
        args.archive,
        args.archive.with_name("code-data-supplement.zip"),
    ]
    for archive in archives:
        if not archive.is_file():
            raise FileNotFoundError(archive)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if member.endswith("/"):
                    continue
                if member.endswith("scripts/audit_submission.py"):
                    continue
                content = bundle.read(member).decode("utf-8", errors="ignore")
                for pattern in patterns:
                    if re.search(pattern, content, flags=re.IGNORECASE):
                        raise ValueError(
                            f"identity pattern {pattern} found in {archive}:{member}"
                        )

    print(
        f"audit passed: {pages} total pages, content ends on page {content_page}, "
        "fonts embedded, anonymous PDF, checklist and archive present"
    )


if __name__ == "__main__":
    main()
