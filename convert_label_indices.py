#!/usr/bin/env python3
"""Replace YOLO label class indices with 0 inside a labels directory.

Also removes any angle field (6th value) so labels keep only the five standard
YOLO bbox values.

Useful when all annotations should belong to a single class but legacy files
still use another index (e.g., `3`). The script rewrites each `.txt` file so the
first value of every non-empty line becomes `0`, preserving the remaining
coordinates (x, y, w, h).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force YOLO label class indices to 0 and remove any angle field."
    )
    default_labels = (
        Path(__file__).resolve().parent
        / "motos_box"
        / "yolo_obb_dataset"
        / "labels"
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=default_labels,
        help=f"Directory containing label splits (default: {default_labels})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would change without modifying them.",
    )
    return parser.parse_args()


def rewrite_file(path: Path, dry_run: bool) -> bool:
    text = path.read_text()
    original_lines = text.splitlines()
    ending_newline = text.endswith("\n")
    updated_lines = []
    changed = False

    for line in original_lines:
        stripped = line.strip()
        if not stripped:
            updated_lines.append(line)
            continue

        parts = stripped.split()
        # força classe = 0
        if parts[0] != "0":
            parts[0] = "0"
            changed = True
        # remove ângulo (6º parâmetro) caso exista
        if len(parts) > 5:
            parts = parts[:5]
            changed = True
        updated_lines.append(" ".join(parts))

    if changed and not dry_run:
        new_text = "\n".join(updated_lines)
        if ending_newline:
            new_text += "\n"
        path.write_text(new_text)
    return changed


def main() -> None:
    args = parse_args()
    labels_dir = args.labels_dir
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {labels_dir}")

    label_files = sorted(labels_dir.rglob("*.txt"))
    if not label_files:
        raise RuntimeError(f"No .txt files found under {labels_dir}")

    changed_count = 0
    for path in label_files:
        if rewrite_file(path, args.dry_run):
            changed_count += 1
            if args.dry_run:
                print(f"Would update {path}")

    verb = "would be" if args.dry_run else "were"
    print(f"{changed_count} label files {verb} updated.")


if __name__ == "__main__":
    main()
