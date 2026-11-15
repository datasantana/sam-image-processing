# Análisis posterior (`python -m segpipe.analyze`)

Corre en CPU, sin torch. Lee el estudio terminado y escribe tablas, figuras y `REPORT.md`.
No modifica nada dentro de `runs/` (lo verifica comparando tamaño y fecha de cada archivo antes y después).

## 1. Obtener el estudio sin checkpoints

Sin Google Drive para escritorio, empaqueta el estudio en Colab (en CPU basta) sin `*.pt`:

```python
from google.colab import drive
drive.mount("/content/drive")
import hashlib, pathlib, shutil, zipfile
src = pathlib.Path("/content/drive/MyDrive/segpipe/pascual_v1")
tmp = pathlib.Path("/content/pascual_v1_light.zip")
n = skipped = 0
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if not p.is_file() or rel.parts[0] == "analysis":
            continue
        if p.suffix == ".pt" or p.name.endswith(".tmp"):
            skipped += 1
            continue
        z.write(p, pathlib.Path("pascual_v1") / rel); n += 1
done = len(list(src.glob("runs/*/seed*/fold*/DONE.json")))
npz = len(list(src.glob("runs/*/seed*/fold*/test/photo_confusion.npz")))
dst = src.parent / tmp.name
shutil.copy(tmp, dst)
print(f"{n} archivos, {skipped} omitidos (*.pt), DONE.json={done}, npz de test={npz}")
print(f"{dst}  {tmp.stat().st_size/1e6:.1f} MB  sha256={hashlib.sha256(tmp.read_bytes()).hexdigest()}")
```

Descarga `MyDrive/segpipe/pascual_v1_light.zip` y descomprímelo fuera del repo.

## 2. Ejecutar

```bash
python -m segpipe.analyze --study <carpeta>/pascual_v1 \
    --manifest <dataset_pack>/manifest.csv \
    --out analysis \
    --mapping configs/analysis/common_scheme.yaml \
    --mauricio <ruta>/Metricas_run05_b3_v2.xlsx
```

- El estudio se busca en este orden: `--study`, la variable `SEGPIPE_STUDY` y por último Google Drive para escritorio (`<unidad>:/Mi unidad/segpipe/pascual_v1` o `My Drive`).
- `--out` va por defecto a `<study>/analysis`; `analysis/` en la raíz del repo está en `.gitignore`.
- A3 solo se calcula si el YAML tiene `approved: true`.

## 3. Salidas

- `REPORT.md`: inventario, verificaciones, método y números de A1–A3, costo y mIoU por pliegue (sin interpretación).
- `tables/`:
  - CSV de cada análisis;
  - `comparacion_recalculada/`: la misma tabla que `segpipe.report`, recalculada y comparada con `results/summary.csv`.
- `figures/`: cada figura en PNG a 300 dpi, en PDF y con un CSV de sus datos.
