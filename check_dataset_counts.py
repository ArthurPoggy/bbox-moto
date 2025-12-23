#!/usr/bin/env python3
from pathlib import Path
import argparse
import math
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Conta imagens/labels e aponta pares faltantes.")
    parser.add_argument(
        "--root",
        default="/home/labgpsi_1/treino-11.12/datasets",
        help="Raiz que contém as pastas images/ e labels/",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Splits separados por vírgula (padrão: train,val,test)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Quantidade de exemplos para mostrar em cada lista",
    )
    parser.add_argument(
        "--check-format",
        action="store_true",
        help="Valida se cada linha de label tem 6 valores (cls x y w h ang) e se estão normalizados",
    )
    parser.add_argument(
        "--inspect-cache",
        action="store_true",
        help="Lê o cache de labels (<root>/labels/<split>.cache) e mostra msgs/nf/nl",
    )
    args = parser.parse_args()

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    root = Path(args.root)

    for split in args.splits.split(","):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in image_exts]
        lbls = list(lbl_dir.glob("*.txt"))
        print(f"{split}: imgs={len(imgs)}, labels={len(lbls)}")

        missing_img = []
        for lbl in lbls:
            stem = lbl.stem
            if not any((img_dir / f"{stem}{ext}").exists() for ext in image_exts):
                missing_img.append(lbl.name)
        if missing_img:
            print(f"  labels sem imagem: {len(missing_img)} (ex.: {missing_img[:args.limit]})")

        lbl_stems = {p.stem for p in lbls}
        missing_lbl = [img.name for img in imgs if img.stem not in lbl_stems]
        if missing_lbl:
            print(f"  imagens sem label: {len(missing_lbl)} (ex.: {missing_lbl[:args.limit]})")

        if args.check_format:
            bad = []
            for lbl in lbls:
                for i, line in enumerate(lbl.read_text().splitlines(), 1):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 6:
                        bad.append((lbl.name, i, "len", len(parts), line))
                        continue
                    cls, *nums = parts
                    if not cls.isdigit():
                        bad.append((lbl.name, i, "cls", cls, line))
                        continue
                    try:
                        x, y, w, h, a = map(float, nums)
                    except Exception as e:  # noqa: BLE001
                        bad.append((lbl.name, i, "float", str(e), line))
                        continue
                    if not all(map(math.isfinite, [x, y, w, h, a])):
                        bad.append((lbl.name, i, "nan", [x, y, w, h, a], line))
                        continue
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        bad.append((lbl.name, i, "range", [x, y, w, h, a], line))
            if bad:
                print(f"  linhas inválidas: {len(bad)} (ex.: {bad[:args.limit]})")

        if args.inspect_cache:
            cache_path = root / "labels" / f"{split}.cache"
            if not cache_path.exists():
                print(f"  cache {cache_path.name}: não encontrado")
            else:
                try:
                    cache = np.load(cache_path, allow_pickle=True).item()
                    msgs = cache.get("msgs", [])
                    nf = cache.get("nf", None)
                    nl = cache.get("nl", None)
                    print(f"  cache {cache_path.name}: nf={nf}, nl={nl}, msgs={len(msgs)}")
                    if msgs:
                        print(f"    msgs: {msgs[:args.limit]}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  cache {cache_path.name}: erro ao ler ({exc})")


if __name__ == "__main__":
    main()
