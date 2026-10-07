"""Check internal links of the wiki.

Every relative Markdown link must point to an existing file, and every
anchor (``page.md#section``) must match a heading of the target page,
using the GitHub heading-slug rules.

Usage:
    python tools/check_links.py

The script exits with status 1 when at least one broken link is found.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORED_PARTS = {".git", "_site", "node_modules", ".venv"}

LINK_PATTERN = re.compile(r"\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_PATTERN = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def fenced_flags(lines: list[str]) -> list[bool]:
    """Flag the lines that belong to fenced code blocks (fence lines included).

    A fence closes only with the same character repeated at least as many
    times as in the opening fence, so a block opened with four backticks can
    contain lines starting with three backticks.
    """
    flags = []
    opening = None
    for line in lines:
        match = FENCE_PATTERN.match(line)
        if opening is None:
            if match:
                opening = match.group(1)
            flags.append(match is not None)
        else:
            flags.append(True)
            if match and match.group(1)[0] == opening[0] and len(match.group(1)) >= len(opening):
                if not line.strip()[len(match.group(1)):].strip():
                    opening = None
    return flags


def strip_code(text: str) -> list[str]:
    """Return the lines of a document with code blocks and code spans blanked out."""
    lines = text.splitlines()
    return [
        "" if in_code else re.sub(r"`[^`]*`", "", line)
        for line, in_code in zip(lines, fenced_flags(lines))
    ]


def github_slug(heading: str) -> str:
    heading = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # keep link text
    heading = heading.replace("`", "")
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\- ]", "", slug)
    return slug.replace(" ", "-")


def anchors_of(path: Path) -> set[str]:
    counts: Counter[str] = Counter()
    anchors = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line, in_code in zip(lines, fenced_flags(lines)):
        if in_code:
            continue
        match = HEADING_PATTERN.match(line)
        if not match:
            continue
        slug = github_slug(match.group(2))
        suffix = f"-{counts[slug]}" if counts[slug] else ""
        counts[slug] += 1
        anchors.add(slug + suffix)
    return anchors


def check() -> int:
    errors = []
    anchor_cache: dict[Path, set[str]] = {}
    for source in markdown_files():
        lines = strip_code(source.read_text(encoding="utf-8"))
        for number, line in enumerate(lines, start=1):
            for target in LINK_PATTERN.findall(line):
                if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                    continue  # external URL, mailto, etc.
                path_part, _, anchor = target.partition("#")
                destination = (source.parent / path_part).resolve() if path_part else source
                location = f"{source.relative_to(ROOT)}:{number}"
                if not destination.exists():
                    errors.append(f"{location}: missing target '{target}'")
                    continue
                if anchor and destination.suffix == ".md":
                    if destination not in anchor_cache:
                        anchor_cache[destination] = anchors_of(destination)
                    if anchor not in anchor_cache[destination]:
                        errors.append(f"{location}: unknown anchor '{target}'")
    for error in errors:
        print(error)
    print(f"Checked {len(markdown_files())} Markdown files: {len(errors)} broken link(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(check())
