"""Esquema de clases desde classes.json y conversión RGB → índice.

El pipeline es *configurable por esquema de clases*: el esquema sale de ``classes.json`` del
dataset, se pueden ignorar clases (p. ej. ``background`` = píxeles sin polígono en CVAT) y se
puede aplicar un mapeo fino → grueso hacia un esquema común (para comparar datasets).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

IGNORE_INDEX = 255


@dataclass
class ClassSchema:
    """Esquema de entrenamiento.

    ``source``: clases del archivo (index, name, rgb) en el orden de classes.json.
    ``names``: clases que predice el modelo (índices contiguos 0..C-1).
    ``source_to_train``: índice fuente → índice de entrenamiento o IGNORE_INDEX.
    """
    source: list[dict]
    names: list[str]
    source_to_train: np.ndarray
    ignored: list[str] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.names)

    @property
    def ignore_index(self) -> int:
        return IGNORE_INDEX

    # ------------------------------------------------------------------ construcción
    @classmethod
    def from_json(cls, path: str | Path, ignore: Optional[list[str]] = None,
                  mapping: Optional[dict[str, str]] = None, target: Optional[list[str]] = None) -> "ClassSchema":
        with open(path, "r", encoding="utf-8") as f:
            source = json.load(f)
        return cls.from_list(source, ignore=ignore, mapping=mapping, target=target)

    @classmethod
    def from_list(cls, source: list[dict], ignore: Optional[list[str]] = None,
                  mapping: Optional[dict[str, str]] = None, target: Optional[list[str]] = None) -> "ClassSchema":
        """``mapping`` (opcional) lleva nombres fuente a nombres de ``target`` (esquema común).

        Sin mapping, el esquema de entrenamiento = clases fuente sin las ignoradas, en el mismo orden.
        Con mapping, toda clase fuente debe estar en ``mapping`` o en ``ignore``.
        """
        ignore = list(ignore or [])
        _check_source(source)
        src_names = [c["name"] for c in source]
        for n in ignore:
            if n not in src_names:
                raise ValueError(f"Clase a ignorar '{n}' no existe en classes.json ({src_names})")
        if mapping:
            if not target:
                target = list(dict.fromkeys(v for k, v in mapping.items() if k not in ignore))
            missing = [n for n in src_names if n not in mapping and n not in ignore]
            if missing:
                raise ValueError(f"Clases sin mapeo al esquema común: {missing}")
            bad = sorted({v for v in mapping.values() if v not in target})
            if bad:
                raise ValueError(f"Destinos de mapeo fuera del esquema común: {bad}")
            names = list(target)
            lut = np.array([IGNORE_INDEX if c["name"] in ignore else names.index(mapping[c["name"]])
                            for c in source], dtype=np.int64)
        else:
            names = [n for n in src_names if n not in ignore]
            lut = np.array([IGNORE_INDEX if c["name"] in ignore else names.index(c["name"]) for c in source],
                           dtype=np.int64)
        return cls(source=source, names=names, source_to_train=lut, ignored=ignore)

    # ------------------------------------------------------------------ conversión
    def rgb_to_index(self, mask_rgb: np.ndarray, where: str = "") -> np.ndarray:
        """Máscara RGB (H, W, 3) en orden **RGB** → índices de entrenamiento (uint8, 255 = ignorar).

        Vectorizado. Falla con ValueError si aparece un color que no está en classes.json.
        """
        if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
            raise ValueError(f"Se esperaba una máscara RGB (H, W, 3), llegó {mask_rgb.shape} {where}")
        m = mask_rgb.astype(np.uint32)
        codes = (m[..., 0] << 16) | (m[..., 1] << 8) | m[..., 2]
        src_codes = np.array([(c["rgb"][0] << 16) | (c["rgb"][1] << 8) | c["rgb"][2] for c in self.source],
                             dtype=np.uint32)
        order = np.argsort(src_codes)
        sorted_codes = src_codes[order]
        pos = np.searchsorted(sorted_codes, codes)
        pos_c = np.clip(pos, 0, len(sorted_codes) - 1)
        known = sorted_codes[pos_c] == codes
        if not known.all():
            bad, cnt = np.unique(codes[~known], return_counts=True)
            desc = ", ".join(f"({int(b) >> 16},{(int(b) >> 8) & 255},{int(b) & 255}): {int(c)} px"
                             for b, c in list(zip(bad, cnt))[:10])
            raise ValueError(f"Colores desconocidos en la máscara {where}: {desc}. "
                             f"Revisa classes.json o la exportación de CVAT.")
        src_idx = order[pos_c]
        out = self.source_to_train[src_idx]
        return out.astype(np.uint8)

    def to_dict(self) -> dict:
        return {"names": self.names, "ignored": self.ignored, "ignore_index": IGNORE_INDEX,
                "source": self.source, "source_to_train": self.source_to_train.tolist()}


def _check_source(source: list[dict]) -> None:
    if not source:
        raise ValueError("classes.json está vacío")
    for i, c in enumerate(source):
        for k in ("index", "name", "rgb"):
            if k not in c:
                raise ValueError(f"classes.json[{i}] no tiene '{k}'")
        if c["index"] != i:
            raise ValueError(f"classes.json no está ordenado: posición {i} tiene index {c['index']}")
        if len(c["rgb"]) != 3:
            raise ValueError(f"classes.json[{i}] rgb inválido: {c['rgb']}")
    names = [c["name"] for c in source]
    if len(set(names)) != len(names):
        raise ValueError("Nombres de clase duplicados en classes.json")
    rgbs = [tuple(c["rgb"]) for c in source]
    if len(set(rgbs)) != len(rgbs):
        raise ValueError("Colores duplicados en classes.json")
