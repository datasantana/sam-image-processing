"""Carga de fotos en RAM, transformaciones y datasets de entrenamiento (resize o recortes nativos).

Aleatoriedad reproducible: cada muestra usa un RNG derivado de (semilla, época, índice), así que
el contenido de una época no depende de cuántas épocas se corrieron antes ni de una reanudación.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .classes import IGNORE_INDEX, ClassSchema


# ----------------------------------------------------------------------------- RAM store

@dataclass
class Photo:
    name: str
    image: np.ndarray            # (H, W, 3) uint8 RGB
    mask: np.ndarray             # (H, W) uint8, índices de entrenamiento, 255 = ignorar
    class_coords: dict = field(default_factory=dict)   # clase → (n, 2) coords (y, x) muestreadas


def load_photo(root: Path, row: dict, schema: ClassSchema, coords_per_class: int = 4000,
               seed: int = 0) -> Photo:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(root / row["image"]) as im:
        img = np.asarray(im.convert("RGB"))
    with Image.open(root / row["mask"]) as im:
        mrgb = np.asarray(im.convert("RGB"))
    mask = schema.rgb_to_index(mrgb, where=row["name"])
    rng = np.random.default_rng(seed)
    coords = {}
    for c in np.unique(mask):
        if c == IGNORE_INDEX:
            continue
        ys, xs = np.nonzero(mask == c)
        k = min(coords_per_class, len(ys))
        sel = rng.choice(len(ys), k, replace=False)
        coords[int(c)] = np.stack([ys[sel], xs[sel]], 1).astype(np.int32)
    return Photo(row["name"], img, mask, coords)


class PhotoStore:
    """Fotos + máscaras convertidas, en RAM (se comparten con los workers vía fork)."""

    def __init__(self, root: str | Path, rows: list[dict], schema: ClassSchema, names: Optional[list[str]] = None,
                 verbose: bool = True):
        self.root = Path(root)
        self.schema = schema
        by_name = {r["name"]: r for r in rows}
        names = names or [r["name"] for r in rows]
        self.photos: dict[str, Photo] = {}
        it = names
        if verbose:
            try:
                from tqdm.auto import tqdm
                it = tqdm(names, desc="cargando fotos")
            except ImportError:
                pass
        for n in it:
            self.photos[n] = load_photo(self.root, by_name[n], schema)

    def __getitem__(self, name: str) -> Photo:
        return self.photos[name]

    def pixel_counts(self, names: list[str]) -> np.ndarray:
        C = self.schema.num_classes
        tot = np.zeros(C, np.int64)
        for n in names:
            m = self.photos[n].mask
            tot += np.bincount(m[m != IGNORE_INDEX].ravel(), minlength=C)[:C]
        return tot

    def ignored_fraction(self) -> dict[str, float]:
        return {n: float((p.mask == IGNORE_INDEX).mean()) for n, p in self.photos.items()}


# ----------------------------------------------------------------------------- transforms

def _try(ctor, *variants):
    """Crea un transform probando firmas de albumentations 2.x y luego 1.x."""
    err = None
    for kw in variants:
        try:
            return ctor(**kw)
        except TypeError as e:  # parámetro no soportado en esta versión
            err = e
    raise err


def build_augment(kind: str, resize: Optional[int] = None):
    """Aumentos geométricos/fotométricos (sin normalizar). ``legacy`` = los del notebook original."""
    import warnings
    import albumentations as A
    warnings.filterwarnings("ignore", message="ShiftScaleRotate is a special case of Affine")
    ops = []
    if resize:
        ops.append(A.Resize(resize, resize))
    if kind == "legacy":
        ops += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            _try(A.ShiftScaleRotate if hasattr(A, "ShiftScaleRotate") else A.Affine,
                 dict(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                 dict(translate_percent=(-0.1, 0.1), scale=(0.8, 1.2), rotate=(-30, 30), p=0.5)),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            ], p=0.5),
            A.OneOf([
                # var_limit (10, 50) en escala 0-255 ≡ std (√10, √50)/255 en albumentations 2.x
                _try(A.GaussNoise, dict(std_range=(10 ** 0.5 / 255, 50 ** 0.5 / 255), p=1.0),
                     dict(var_limit=(10, 50), p=1.0)),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
            # 1.x: max_holes=8 (min_holes=None ⇒ 8), 16×16 px
            _try(A.CoarseDropout,
                 dict(num_holes_range=(8, 8), hole_height_range=(16, 16), hole_width_range=(16, 16),
                      fill_mask=IGNORE_INDEX, p=0.1),
                 dict(max_holes=8, max_height=16, max_width=16, mask_fill_value=IGNORE_INDEX, p=0.1)),
        ]
    elif kind == "hflip":
        ops.append(A.HorizontalFlip(p=0.5))
    elif kind != "none":
        raise ValueError(f"augment desconocido: {kind}")
    return A.Compose(ops)


def seed_compose(compose, seed: int) -> None:
    """Fija la aleatoriedad de un Compose de albumentations para una muestra."""
    if hasattr(compose, "set_random_seed"):
        compose.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


def to_tensor(img: np.ndarray, mask: Optional[np.ndarray], mean, std):
    import torch
    img = np.ascontiguousarray(img)
    if not img.flags.writeable:
        img = img.copy()
    x = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
    x = (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    if mask is None:
        return x
    return x, torch.from_numpy(np.ascontiguousarray(mask)).long()


def _sample_seed(seed: int, epoch: int, idx: int) -> int:
    return int(np.random.SeedSequence([seed, epoch, idx]).generate_state(1)[0])


# ----------------------------------------------------------------------------- datasets

class _EpochDataset:
    epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class ResizeDataset(_EpochDataset):
    """J0: cada foto completa, reescalada a ``size``×``size`` + aumentos. Una época = una pasada."""

    def __init__(self, store: PhotoStore, names: list[str], data_cfg: dict, seed: int):
        self.store, self.names, self.cfg, self.seed = store, list(names), data_cfg, seed
        self.aug = build_augment(data_cfg["augment"], resize=data_cfg["resize"])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        p = self.store[self.names[i]]
        seed_compose(self.aug, _sample_seed(self.seed, self.epoch, i))
        out = self.aug(image=p.image, mask=p.mask)
        return to_tensor(out["image"], out["mask"], self.cfg["mean"], self.cfg["std"])


class CropDataset(_EpochDataset):
    """J1+: recortes aleatorios a resolución nativa (opcionalmente con reescalado y sesgo a clases raras)."""

    def __init__(self, store: PhotoStore, names: list[str], data_cfg: dict, seed: int,
                 rare_classes: Optional[list[int]] = None):
        import cv2  # noqa: F401  (usado en _scaled_crop)
        self.store, self.names, self.cfg, self.seed = store, list(names), data_cfg, seed
        self.crop = int(data_cfg["crop"])
        self.n = int(data_cfg.get("samples_per_epoch") or len(self.names))
        self.aug = build_augment(data_cfg["augment"])
        self.rare = list(rare_classes or [])
        self.rare_photos = {c: [n for n in self.names if c in store[n].class_coords] for c in self.rare}
        self.rare = [c for c in self.rare if self.rare_photos[c]]

    def __len__(self):
        return self.n

    def _scaled_crop(self, img, mask, cy, cx, side, rng):
        """Recorta un cuadrado de ``side`` px nativos centrado (aprox.) en (cy, cx) y lo lleva a crop×crop."""
        import cv2
        H, W = mask.shape
        side = int(round(side))
        y0 = int(np.clip(cy - side // 2, 0, max(0, H - side)))
        x0 = int(np.clip(cx - side // 2, 0, max(0, W - side)))
        im = img[y0:y0 + side, x0:x0 + side]
        mk = mask[y0:y0 + side, x0:x0 + side]
        ph, pw = side - im.shape[0], side - im.shape[1]
        if ph > 0 or pw > 0:  # foto más pequeña que el recorte
            im = np.pad(im, ((0, ph), (0, pw), (0, 0)))
            mk = np.pad(mk, ((0, ph), (0, pw)), constant_values=IGNORE_INDEX)
        if side != self.crop:
            im = cv2.resize(im, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)
            mk = cv2.resize(mk, (self.crop, self.crop), interpolation=cv2.INTER_NEAREST)
        return im, mk

    def __getitem__(self, i):
        s = _sample_seed(self.seed, self.epoch, i)
        rng = np.random.default_rng(s)
        sr = self.cfg.get("scale_range")
        scale = float(rng.uniform(sr[0], sr[1])) if sr else 1.0
        side = self.crop / scale
        if self.rare and self.cfg.get("sampler") == "rare" and rng.random() < float(self.cfg["rare_p"]):
            c = self.rare[rng.integers(len(self.rare))]
            p = self.store[self.rare_photos[c][rng.integers(len(self.rare_photos[c]))]]
            yx = p.class_coords[c][rng.integers(len(p.class_coords[c]))]
            jit = rng.integers(-int(side) // 4, int(side) // 4 + 1, size=2)
            cy, cx = int(yx[0] + jit[0]), int(yx[1] + jit[1])
        else:
            p = self.store[self.names[rng.integers(len(self.names))]]
            H, W = p.mask.shape
            cy = int(rng.integers(0, H)); cx = int(rng.integers(0, W))
        im, mk = self._scaled_crop(p.image, p.mask, cy, cx, side, rng)
        seed_compose(self.aug, s)
        out = self.aug(image=im, mask=mk)
        return to_tensor(out["image"], out["mask"], self.cfg["mean"], self.cfg["std"])


def rare_classes_auto(pixel_counts: np.ndarray, frac: float) -> list[int]:
    tot = pixel_counts.sum()
    return [int(c) for c in np.nonzero((pixel_counts > 0) & (pixel_counts < frac * tot))[0]]


def build_train_dataset(store: PhotoStore, names: list[str], data_cfg: dict, seed: int, class_names: list[str]):
    if data_cfg["mode"] == "resize":
        return ResizeDataset(store, names, data_cfg, seed), []
    if data_cfg["mode"] != "crops":
        raise ValueError(f"data.mode desconocido: {data_cfg['mode']}")
    rare: list[int] = []
    if data_cfg.get("sampler") == "rare":
        rc = data_cfg.get("rare_classes", "auto")
        if rc == "auto":
            rare = rare_classes_auto(store.pixel_counts(names), float(data_cfg.get("rare_px_frac", 0.01)))
        else:
            rare = [class_names.index(n) for n in rc]
    elif data_cfg.get("sampler") not in (None, "uniform"):
        raise ValueError(f"data.sampler desconocido: {data_cfg['sampler']}")
    return CropDataset(store, names, data_cfg, seed, rare), rare
