"""
Converte labels YOLO retangulares (class x_center y_center width height)
para o formato poligonal (class x1 x2 x3 x4 y1 y2 y3 y4), salvando a saída
na subpasta "converted" dentro de box_motos para preservar os arquivos originais.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Iterable, List, Tuple


# Usamos Decimal para evitar perdas de precisão e não arredondar manualmente.
getcontext().prec = 28

# Caminhos conhecidos (Windows e WSL). O primeiro existente será utilizado.
KNOWN_BOX_MOTOS_DIRS: Tuple[Path, ...] = (
    Path(r"C:\OCR-PLACAS\BBOX-MOTO\box_motos"),
    Path("/mnt/c/OCR-PLACAS/BBOX-MOTO/box_motos"),
)


def resolve_box_motos_dir() -> Path:
    """Retorna o diretório box_motos conhecido ou infere relativo ao script."""
    for candidate in KNOWN_BOX_MOTOS_DIRS:
        if candidate.exists():
            return candidate
    local_candidate = Path(__file__).resolve().parent / "box_motos"
    if local_candidate.exists():
        return local_candidate
    raise FileNotFoundError(
        "Não foi possível localizar o diretório box_motos. "
        "Atualize KNOWN_BOX_MOTOS_DIRS ou crie a pasta esperada."
    )


def parse_line(parts: List[str]) -> Tuple[str, Decimal, Decimal, Decimal, Decimal]:
    """Converte tokens em Decimal preservando a classe como string."""
    label = parts[0]
    try:
        x_center = Decimal(parts[1])
        y_center = Decimal(parts[2])
        width = Decimal(parts[3])
        height = Decimal(parts[4])
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Não foi possível converter valores numéricos: {parts}") from exc
    return label, x_center, y_center, width, height


def to_polygon(
    label: str, x_center: Decimal, y_center: Decimal, width: Decimal, height: Decimal
) -> str:
    """Calcula os vértices do retângulo a partir do centro e dimensões."""
    half_w = width / Decimal(2)
    half_h = height / Decimal(2)
    x_min = x_center - half_w
    x_max = x_center + half_w
    y_min = y_center - half_h
    y_max = y_center + half_h

    # Ordem: topo-esquerda, topo-direita, baixo-direita, baixo-esquerda.
    values: Iterable[Decimal | str] = (
        label,
        x_min,
        x_max,
        x_max,
        x_min,
        y_min,
        y_min,
        y_max,
        y_max,
    )
    return " ".join(str(value) for value in values)


def convert_file(source: Path, destination_dir: Path) -> None:
    """Converte um .txt e grava o resultado em destination_dir."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    converted_lines: List[str] = []

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 5:
                print(
                    f"[AVISO] {source.name}: linha {line_number} ignorada "
                    f"(esperado 5 colunas, recebido {len(parts)})."
                )
                continue
            try:
                label, xc, yc, w, h = parse_line(parts)
            except ValueError as err:
                print(f"[AVISO] {source.name}: linha {line_number} ignorada ({err}).")
                continue
            converted_lines.append(to_polygon(label, xc, yc, w, h))

    destination.write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8")


def main() -> None:
    box_motos_dir = resolve_box_motos_dir()
    converted_dir = box_motos_dir / "converted"
    txt_files = sorted(file for file in box_motos_dir.glob("*.txt") if file.is_file())
    if not txt_files:
        print(f"Nenhum arquivo .txt encontrado em {box_motos_dir}")
        return

    for txt_file in txt_files:
        convert_file(txt_file, converted_dir)

    print(
        f"{len(txt_files)} arquivo(s) processado(s) com sucesso. "
        f"Resultados salvos em {converted_dir}"
    )


if __name__ == "__main__":
    main()
