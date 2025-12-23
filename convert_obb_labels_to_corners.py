#!/usr/bin/env python3
"""Converte labels YOLO OBB (cx, cy, w, h, angle) para 4 vértices explicitados.

Cada linha `class cx cy w h angle` é reescrita para
`class x1 y1 x2 y2 x3 y3 x4 y4`, mantendo os valores normalizados.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reescreve labels YOLO OBB para formato com 4 vértices.")
    default_labels = Path(__file__).resolve().parent / "motos_box" / "box_motos"
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=default_labels,
        help=f"Pasta com splits de labels (default: {default_labels})",
    )
    parser.add_argument(
        "--angle-format",
        choices=["radians", "degrees"],
        default="radians",
        help="Unidade dos ângulos armazenados nos labels originais.",
    )
    parser.add_argument(
        "--default-angle",
        type=float,
        default=0.0,
        help="Ângulo usado quando o label tiver apenas 5 valores (sem ângulo).",
    )
    parser.add_argument(
        "--output-order",
        choices=["grouped", "interleaved"],
        default="grouped",
        help="grouped = x1 x2 x3 x4 y1 y2 y3 y4, interleaved = x1 y1 ...",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra quais arquivos seriam alterados sem sobrescrevê-los.",
    )
    return parser.parse_args()


def format_float(value: float) -> str:
    text = f"{value:.15f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def rotate_points(
    cx: float, cy: float, w: float, h: float, angle_rad: float
) -> list[tuple[float, float]]:
    half_w = w / 2.0
    half_h = h / 2.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Vértices no retângulo alinhado antes da rotação (sentido horário).
    rel_points = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]

    corners: list[tuple[float, float]] = []
    for px, py in rel_points:
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        corners.append((cx + rx, cy + ry))
    return corners


def convert_line(parts: list[str], angle_format: str, default_angle: float, order: str) -> str:
    if len(parts) < 5:
        raise ValueError("Linha não possui 5 campos obrigatórios.")
    if len(parts) == 9:
        # Já parece estar no formato alvo.
        return " ".join(parts)

    cls = parts[0]
    try:
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        angle = float(parts[5]) if len(parts) >= 6 else default_angle
    except ValueError as exc:  # pragma: no cover - input inválido
        raise ValueError(f"Não foi possível interpretar floats: {' '.join(parts)}") from exc

    angle_rad = math.radians(angle) if angle_format == "degrees" else angle
    corners = rotate_points(cx, cy, w, h, angle_rad)

    flattened: list[str] = [cls]
    if order == "grouped":
        xs = [format_float(x) for x, _ in corners]
        ys = [format_float(y) for _, y in corners]
        flattened.extend(xs)
        flattened.extend(ys)
    else:
        for x, y in corners:
            flattened.append(format_float(x))
            flattened.append(format_float(y))
    return " ".join(flattened)


def iter_label_files(root: Path) -> Iterable[Path]:
    return sorted(root.rglob("*.txt"))


def process_file(path: Path, angle_format: str, default_angle: float, order: str, dry_run: bool) -> bool:
    text = path.read_text()
    lines = text.splitlines()
    ending_newline = text.endswith("\n")

    new_lines = []
    changed = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        try:
            converted = convert_line(parts, angle_format, default_angle, order)
        except ValueError as err:
            raise ValueError(f"Erro no arquivo {path}: {err}") from err
        if converted != stripped:
            changed = True
        new_lines.append(converted)

    if changed:
        if dry_run:
            print(f"[dry-run] Atualizaria {path}")
        else:
            new_text = "\n".join(new_lines)
            if ending_newline and new_lines:
                new_text += "\n"
            path.write_text(new_text)
    return changed


def main() -> None:
    args = parse_args()
    labels_dir = args.labels_dir
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {labels_dir}")

    files = list(iter_label_files(labels_dir))
    if not files:
        raise RuntimeError(f"Nenhum arquivo .txt encontrado em {labels_dir}")

    changed = 0
    for path in files:
        if process_file(path, args.angle_format, args.default_angle, args.output_order, args.dry_run):
            changed += 1

    verb = "seriam" if args.dry_run else "foram"
    print(f"{changed} arquivos {verb} atualizados.")


if __name__ == "__main__":
    main()
