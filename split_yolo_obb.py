#!/usr/bin/env python3
"""Split motos_box dataset into YOLO-OBB train/val/test folders.

The script expects the following layout relative to the repository root:
```
motos_box/
├── box_motos        # label files (*.txt)
└── imgs_com_box     # image files (*.jpg|*.jpeg|*.png)
```
It also supports datasets structured as:
```
dataset/
├── images
└── labels
```
It creates `motos_box/yolo_obb_dataset/{images,labels}/{train,val,test}` with a
70/20/10 split by default, copying paired image/label files into each split.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split imgs_com_box/box_motos into YOLO-OBB train/val/test folders."
        )
    )
    default_base = Path(__file__).resolve().parent / "motos_box"
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base,
        help=f"Base path containing imgs_com_box and box_motos (default: {default_base})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory (default: <base-dir>/yolo_obb_dataset)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Explicit path to the directory that stores all images.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=None,
        help="Explicit path to the directory that stores label txt files.",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.2, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Fractions for train/val/test (must sum to 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling pairs before splitting",
    )
    return parser.parse_args()


def resolve_data_dirs(base_dir: Path, image_dir: Path | None, label_dir: Path | None) -> Tuple[Path, Path]:
    if image_dir and label_dir:
        return image_dir, label_dir

    candidates = [
        ("imgs_com_box", "box_motos"),
        ("images", "labels"),
    ]
    for img_name, lbl_name in candidates:
        img_path = base_dir / img_name
        lbl_path = base_dir / lbl_name
        if img_path.is_dir() and lbl_path.is_dir():
            return img_path, lbl_path

    raise FileNotFoundError(
        f"Could not locate image/label folders inside {base_dir}. "
        "Expected either ('imgs_com_box', 'box_motos') or ('images', 'labels'), "
        "or specify --image-dir/--label-dir explicitly."
    )


def gather_pairs(
    image_dir: Path, label_dir: Path, image_exts: Sequence[str]
) -> List[Tuple[Path, Path]]:
    images: Dict[str, Path] = {}
    for ext in image_exts:
        for img_path in image_dir.rglob(f"*{ext}"):
            images[img_path.stem] = img_path

    labels = {path.stem: path for path in label_dir.rglob("*.txt")}
    common_ids = sorted(images.keys() & labels.keys())
    if not common_ids:
        raise RuntimeError(
            f"No matching image/label pairs found in {image_dir} and {label_dir}"
        )

    missing_images = sorted(labels.keys() - images.keys())
    missing_labels = sorted(images.keys() - labels.keys())
    if missing_images:
        print(f"Warning: {len(missing_images)} labels lack images (examples: {missing_images[:5]})")
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images lack labels (examples: {missing_labels[:5]})")

    return [(images[_id], labels[_id]) for _id in common_ids]


def ensure_dirs(output_dir: Path, splits: Sequence[str]) -> Dict[str, Dict[str, Path]]:
    structure: Dict[str, Dict[str, Path]] = {}
    for split in splits:
        img_split = output_dir / "images" / split
        lbl_split = output_dir / "labels" / split
        img_split.mkdir(parents=True, exist_ok=True)
        lbl_split.mkdir(parents=True, exist_ok=True)
        structure[split] = {"images": img_split, "labels": lbl_split}
    return structure


def compute_split_indices(total: int, ratios: Sequence[float]) -> List[int]:
    indices: List[int] = []
    start = 0
    for ratio in ratios[:-1]:
        count = math.floor(total * ratio)
        indices.append(start + count)
        start += count
    indices.append(total)
    return indices


def split_dataset(pairs: List[Tuple[Path, Path]], ratios: Sequence[float], seed: int) -> Dict[str, List[Tuple[Path, Path]]]:
    if len(ratios) != 3:
        raise ValueError("Expected three split ratios for train/val/test.")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, received {ratios}.")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    total = len(pairs)
    train_end, val_end, _ = compute_split_indices(total, ratios)
    split_data = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }
    return split_data


def copy_pairs(pairs: Iterable[Tuple[Path, Path]], destinations: Dict[str, Path]) -> None:
    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, destinations["images"] / img_path.name)
        shutil.copy2(lbl_path, destinations["labels"] / lbl_path.name)


def main() -> None:
    args = parse_args()

    base_dir = args.base_dir
    image_dir, label_dir = resolve_data_dirs(base_dir, args.image_dir, args.label_dir)
    output_dir = args.output_dir or (base_dir / "yolo_obb_dataset")

    pairs = gather_pairs(image_dir, label_dir, image_exts=(".jpg", ".jpeg", ".png"))
    split_data = split_dataset(pairs, args.split_ratios, args.seed)
    destinations = ensure_dirs(output_dir, split_data.keys())

    for split, split_pairs in split_data.items():
        copy_pairs(split_pairs, destinations[split])
        print(f"{split}: copied {len(split_pairs)} pairs.")

    print(f"Finished. Output dataset at {output_dir}")


if __name__ == "__main__":
    main()
