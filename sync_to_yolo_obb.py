#!/usr/bin/env python3
"""Sincroniza imagens/labels das pastas fonte para o dataset YOLO OBB.

- Compara os stems entre:
    imagens fonte  : /mnt/c/OCR-PLACAS/BBOX-MOTO/motos_box/imgs_com_box
    labels fonte   : /mnt/c/OCR-PLACAS/BBOX-MOTO/motos_box/box_motos
    dataset alvo   : /mnt/c/OCR-PLACAS/BBOX-MOTO/motos_box/yolo_obb_dataset (images/ e labels/ com splits)
- Copia o que estiver faltando para o dataset alvo, colocando no split indicado (padrão: train).
- Ao salvar labels, força classe = 0 e garante ângulo (6º valor) como float (`0.0` por padrão).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sincroniza imagens/labels fontes para o dataset YOLO OBB.")
    base = Path(__file__).resolve().parent / "motos_box"
    parser.add_argument(
        "--source-images",
        type=Path,
        default=base / "imgs_com_box",
        help="Pasta com imagens fonte.",
    )
    parser.add_argument(
        "--source-labels",
        type=Path,
        default=base / "box_motos",
        help="Pasta com labels fonte.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=base / "yolo_obb_dataset",
        help="Raiz do dataset YOLO OBB (contém images/ e labels/).",
    )
    parser.add_argument(
        "--default-split",
        choices=["train", "val", "test"],
        default="train",
        help="Split onde serão colocados arquivos novos quando não houver split prévio.",
    )
    parser.add_argument(
        "--angle",
        default="0.0",
        help="Valor do ângulo para labels OBB (6º campo).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o que seria copiado/alterado sem gravar arquivos.",
    )
    return parser.parse_args()


def map_stems_by_split(root: Path, kind: str) -> dict[str, str]:
    """Retorna stem -> split para imagens ou labels existentes no dataset alvo."""
    mapping: dict[str, str] = {}
    for split in ["train", "val", "test"]:
        base = root / kind / split
        if not base.exists():
            continue
        if kind == "images":
            files = [p for p in base.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        else:
            files = list(base.glob("*.txt"))
        for p in files:
            mapping[p.stem] = split
    return mapping


def load_source_files(images_dir: Path, labels_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    imgs = {p.stem: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    lbls = {p.stem: p for p in labels_dir.glob("*.txt")}
    return imgs, lbls


def normalize_label(text: str, angle_value: str) -> str:
    lines = text.splitlines()
    ending_newline = text.endswith("\n")
    out = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        # força classe = 0
        if parts:
            parts[0] = "0"
        # garante ângulo (6º valor) como float/texto fornecido
        if len(parts) == 5:
            parts.append(angle_value)
        elif len(parts) >= 6:
            parts[5] = angle_value
        out.append(" ".join(parts))
    result = "\n".join(out)
    if ending_newline and out:
        result += "\n"
    return result


def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def copy_missing(args: argparse.Namespace) -> None:
    source_imgs, source_lbls = load_source_files(args.source_images, args.source_labels)
    target_img_split = map_stems_by_split(args.target, "images")
    target_lbl_split = map_stems_by_split(args.target, "labels")

    all_stems = set(source_imgs) | set(source_lbls)
    added_imgs = added_lbls = 0

    for stem in sorted(all_stems):
        split = target_img_split.get(stem) or target_lbl_split.get(stem) or args.default_split

        # imagem
        if stem not in target_img_split and stem in source_imgs:
            src_img = source_imgs[stem]
            dst_img = args.target / "images" / split / src_img.name
            ensure_dir(dst_img.parent, args.dry_run)
            if args.dry_run:
                print(f"[dry-run] copiar imagem {src_img} -> {dst_img}")
            else:
                shutil.copy2(src_img, dst_img)
            added_imgs += 1

        # label
        if stem not in target_lbl_split and stem in source_lbls:
            src_lbl = source_lbls[stem]
            dst_lbl = args.target / "labels" / split / src_lbl.name
            ensure_dir(dst_lbl.parent, args.dry_run)
            normalized = normalize_label(src_lbl.read_text(), args.angle)
            if args.dry_run:
                print(f"[dry-run] copiar label {src_lbl} -> {dst_lbl}")
            else:
                dst_lbl.write_text(normalized)
            added_lbls += 1

    print(f"Imagens adicionadas: {added_imgs}")
    print(f"Labels adicionadas: {added_lbls}")


def main() -> None:
    args = parse_args()
    copy_missing(args)


if __name__ == "__main__":
    main()
