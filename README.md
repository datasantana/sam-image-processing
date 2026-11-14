# SAM Image Processing — segmentación semántica de imágenes UAV oblicuas

Primera aproximación del proyecto de grado (MSc MGITG): segmentación semántica de fotos oblicuas de dron. El repo contiene:

- **`segpipe/`** — *pipeline configurable por esquema de clases*: obtención y validación del dataset, split espacial por bloques, entrenamiento con checkpoints y reanudación, métricas con IC por bootstrap y tabla comparativa. Probado con `pytest` en CPU.
- **`notebooks/train_ablation.ipynb`** — orquesta en Colab la escalera de ablación J0–J3 y la variante M-b3 (receta de un compañero).
- **`notebooks/automatic_mask_generator_example.ipynb`** — SAM 2 para pre-anotar fotos y exportar COCO a CVAT.
- **`notebooks/legacy/semantic_segmentation_training.ipynb`** — notebook original, sin cambios (registro histórico; usaba Azure Blob Storage, que ya no existe).

---

## Estructura

```
sam-image-processing/
├── segpipe/
│   ├── data_io.py     # descarga (Drive API / Drive montado / local), sha256, extracción, validación
│   ├── classes.py     # esquema desde classes.json, clases ignoradas, mapeo fino→grueso, RGB→índice
│   ├── split.py       # centro de huella oblicua, bloques espaciales, k pliegues con buffer
│   ├── data.py        # fotos en RAM, aumentos, datasets Resize (J0) y recortes nativos (J1+)
│   ├── models.py      # DeepLabv3+, PSPNet, UNet++ (smp) y SegFormer (transformers)
│   ├── losses.py      # Dice + CE ponderada, CE, focal; ignore_index
│   ├── infer.py       # inferencia resize o ventana deslizante → logits a resolución nativa
│   ├── metrics.py     # matriz de confusión, mIoU/OA/mAcc/mF1, clases no evaluables, bootstrap
│   ├── train.py       # una corrida: AMP, acumulación, checkpoint por época, reanudación
│   ├── runner.py      # estudio (dataset + split) y cola config × semilla × pliegue
│   ├── report.py      # tabla comparativa entre configs
│   ├── config.py      # configs YAML con herencia (base:)
│   └── env.py         # GPU, micro-lote, workers, versiones, commit
├── configs/
│   ├── datasets/pascual.yaml       # file ID, sha256, rutas, clases ignoradas, columnas de geometría
│   └── runs/J0.yaml … J3.yaml, M-b3.yaml (+ J2-loss, PSPNet, UNetPP opcionales)
├── notebooks/  (train_ablation, automatic_mask_generator_example, legacy/)
├── tools/pack_dataset.py           # construye el segpack desde la exportación de CVAT
├── docs/  (smoke_run.md, mauricio_config.md)
├── tests/                          # pytest en CPU con datos sintéticos
├── requirements.txt  pyproject.toml  INSTRUCTIONS.md
```

---

## Dataset

`segpack_v1.zip` (560 108 771 bytes, sha256 `1ef52bcb49ef74c05d547f16b042cb9de2371cdd05ddb7119bc84d2c196f41ab`):

| Contenido | |
|---|---|
| `images/<base>.JPG` | 66 fotos oblicuas 4032×2268 (gimbal ≈ −45°, altura relativa ≈ 58.7 m; vuelos del 2025-05-22 y 2025-06-11) |
| `masks/<base>.png` | máscaras RGB de CVAT |
| `classes.json` | lista ordenada `index, name, rgb`: **fuente de verdad** del esquema |
| `manifest.csv` | tamaño, sha256 y metadatos DJI (GPS, altura, gimbal) por foto |
| `class_stats.csv`, `labelmap.txt`, `REPORT.txt` | píxeles por clase, labelmap de CVAT, reporte de empaquetado |

Clases del modelo (12): Brownfields, Buildings, Fence, Ground Surface, Lightning Posts, Parkinglots, Paved Surface, Pools, Road Surface, Sports Fields, Vegetation, Vehicle. **`background` (negro) = píxeles sin polígono en CVAT**: se ignora (`ignore_index`) en la pérdida, las métricas y la matriz de confusión. Es una corrección respecto al notebook original, que lo entrenaba como clase (y, por un bug en la lectura del labelmap, usaba 14 clases; esos checkpoints no son compatibles). El % de píxeles ignorados por foto se reporta en `STUDY_DIR/dataset/ignored_px_per_photo.csv` (en el pack: mediana 22.8 %, rango 8.9–51.4 %).

**Dónde está:** Google Drive (archivo privado, file ID en `configs/datasets/pascual.yaml`). Hay un respaldo manual en OneDrive cuyo enlace guarda el autor (no está automatizado). Para reconstruirlo desde una exportación de CVAT: `python tools/pack_dataset.py --images DIR --cvat-zip masks.zip --out segpack_v1.zip`.

**Cómo se obtiene (`segpipe.data_io.ensure_dataset`)**, en este orden:
1. Copia ya descomprimida y validada en `/content/data/segpack_v1`.
2. Colab: descarga por file ID **con tu cuenta** (`google.colab.auth.authenticate_user` + Drive API `files().get_media` por rangos de 32 MB, reanudable, con barra de progreso). Funciona con el archivo privado y no depende de gdown ni de enlaces públicos.
3. Drive montado en `/content/drive` (ruta configurable `archive.drive_mounted_path`). El zip se copia al disco local.
4. Local (solo pruebas): `SEGPACK_DIR` (carpeta descomprimida; se verifican los sha256 de las imágenes) o `SEGPACK_ZIP`.

Siempre se verifica el tamaño y el sha256 del zip, se descomprime al disco local de la VM y se valida contra `manifest.csv` (parejas completas, tamaños de imagen y máscara, archivos sobrantes). Durante el entrenamiento nunca se lee de Drive.

---

## Protocolo de evaluación (siempre activo)

- **Split espacial por bloques.** Cada foto se representa por el centro aproximado de su huella en tierra, no por la posición del dron: `d = RelativeAltitude / tan(|GimbalPitch|)` (≈ 59 m) en la dirección de `GimbalYawDegree`. Los centros se agrupan en una grilla de **150 m** (≈ tamaño de una huella). Los bloques completos se reparten en **k = 5 pliegues**, balanceando el nº de fotos y la presencia de clases raras. En el pliegue *f*: test = *f*, val = *f*+1, train = el resto **menos las fotos a < 50 m** (buffer) de cualquier foto de val o test. Con el pack real quedan 31–39 fotos de train, 12–16 de val y 12–16 de test por pliegue; se excluyen 2–7 por el buffer.
  - Limitación: la huella oblicua mide ≈ 120 m de fondo. Un buffer de 50 m quita las fotos casi duplicadas, pero no todo el solape: eliminarlo dejaría casi sin train. La distancia mínima train↔val/test de cada pliegue queda en el reporte del split.
  - Con 66 fotos un split fijo es frágil: con val y test de ~13 fotos, varias clases quedarían "no evaluables". Por eso el protocolo del artículo es **CV por bloques**: al juntar el test de todos los pliegues, cada foto se evalúa exactamente una vez y todas las clases tienen ≥ 12 fotos de test. `PROTOCOL="fixed"` usa solo el pliegue 0.
  - Las listas de cada partición se guardan en `STUDY_DIR/splits/` (`splits.json`, `fold<k>.csv`, `split_report.txt`).
- **Métricas** desde matrices de confusión por foto a **resolución nativa**: mIoU, OA, mAcc, mF1 y métricas por clase (IoU, recall, precisión, F1). Una clase presente en < **K = 5** fotos de la partición evaluada se marca **"no evaluable"** y se excluye de mIoU/mAcc/mF1, y queda explícito en la salida. **IC 95 % por bootstrap remuestreando fotos** (2000 réplicas). Salida JSON + CSV por corrida.
- La selección del checkpoint y el early stopping usan solo val. Test se evalúa una vez, con `best.pt`.

---

## Escalera de ablación

Cada paso cambia una sola cosa respecto al anterior (`segpipe.runner.describe_ladder()` imprime el diff exacto de las configs).

| Config | Cambio | Detalle |
|---|---|---|
| **J0** | receta original | DeepLabv3+ ResNet-50 (smp), Resize 512×512, aumentos originales (flips, rot90, ShiftScaleRotate, color, ruido/blur, CoarseDropout 8×16 px), Dice + CE ponderada (1/frecuencia, media 1), Adam lr 1e-4 wd 1e-4, ReduceLROnPlateau (×0.5, paciencia 5), batch 8, 50 épocas, early stopping 10 sobre mIoU de val. Predice a 512 y reescala los logits a 4032×2268. |
| **J1** | resolución | Recortes aleatorios de **1024 px nativos** (880 por época) e inferencia por **ventana deslizante** 1024 con 50 % de solape. Hipótesis a medir: el resize (~3–4.5 → ~20–25 cm/px) borra clases delgadas (Fence, Lightning Posts). |
| **J2** | clases raras | Con p = 0.5 el recorte se centra en un píxel de una clase rara (< 1 % de los píxeles de train). La pérdida no cambia. Alternativa de pérdida opcional: `J2-loss` (Dice + focal, sin pesos). |
| **J3** | modelo | SegFormer **mit-b3** (`nvidia/mit-b3`, transformers); el resto es idéntico a J2. |
| **M-b3** | aparte | Receta de Mauricio (aprox.) sobre nuestro dataset y protocolo; ver [docs/mauricio_config.md](docs/mauricio_config.md) para lo tomado de su hoja Config y lo supuesto. |

También quedan registrados **PSPNet** y **UNet++** (`configs/runs/PSPNet.yaml`, `UNetPP.yaml`: J2 con otro modelo), fuera de la cola por defecto.

---

## Cómo correr en Colab

### Desde el navegador

1. Abre `notebooks/train_ablation.ipynb` con el badge de Colab (o *Archivo → Abrir cuaderno → GitHub*). Elige GPU (A100 o L4).
2. **Sección 1:** clona el repo en `/content/sam-image-processing` e instala solo los paquetes que falten. Se usa el torch que trae Colab, sin reinstalarlo ni fijar `cu118`.
3. **Sección 2:** monta Drive y define `STUDY_DIR` (en Drive), `PROTOCOL`, `CONFIGS` y `SEEDS`.
4. **Sección 3:** obtiene y valida el dataset (pide autorizar tu cuenta la primera vez), construye o reutiliza el split y reporta los píxeles ignorados.
5. **Sección 4 (recomendado la primera vez):** corrida de humo; ver [docs/smoke_run.md](docs/smoke_run.md).
6. **Sección 5:** cola de corridas. **Si la sesión se corta, vuelve a ejecutar las secciones 1–3 y la misma celda de la sección 5**: cada corrida continúa desde su último `last.pt` y las terminadas se saltan.
7. **Sección 6:** tabla comparativa para el artículo.

### Desde VS Code

Con la extensión de Colab para VS Code (*Select Kernel → Colab → New Colab Server*, eligiendo GPU), abre el mismo notebook; el kernel corre en la VM de Google, no en tu PC. Las secciones son las mismas. **No probado:** si `authenticate_user()` o `drive.mount()` no funcionan en la extensión, `ensure_dataset` pasa a la siguiente fuente. Como último recurso, sube el zip a la VM y define `os.environ["SEGPACK_ZIP"] = "/content/segpack_v1.zip"`. Los checkpoints necesitan una ruta en Drive montado para sobrevivir a la sesión.

### Recursos y tiempo

GPU, micro-lote y workers se detectan automáticamente. El batch efectivo de cada config se mantiene con acumulación de gradiente si no cabe en la GPU. Se usa AMP (bf16 si la GPU lo soporta; si no, fp16 con GradScaler). Puedes forzar el micro-lote con `train.micro_batch` en el YAML. La cola completa con CV es 5 configs × 3 semillas × 5 pliegues = **75 corridas**: planifica varias sesiones y usa `MAX_RUNS` para limitar cada una.

---

## Cómo reproducir J0–J3

```python
from segpipe.runner import prepare_study, run_queue
from segpipe import report
study = prepare_study("/content/drive/MyDrive/segpipe/pascual_v1", "configs/datasets/pascual.yaml",
                      protocol="cv", k=5, block_m=150, buffer_m=50, split_seed=0)
run_queue(study, ["J0", "J1", "J2", "J3", "M-b3"], seeds=[42, 43, 44])
res = report.build("/content/drive/MyDrive/segpipe/pascual_v1", ["J0", "J1", "J2", "J3", "M-b3"])
```

Cada corrida (`STUDY_DIR/runs/<config>/seed<s>/fold<f>/`) guarda: `config.json` (config resuelta), `split.json`, `env.json` (GPU, versiones de librerías, commit de git y estado *dirty*, una entrada por sesión), `epoch_log.csv` (pérdida, mIoU/min-IoU/OA de val, IoU por clase, lr, segundos), `best.pt`, `val/` y `test/` (`metrics.json`, `metrics_per_class.csv`, `photo_confusion.npz`) y `DONE.json`. `last.pt` (modelo, optimizador, scheduler, scaler, época, mejor métrica, paciencia, historial y estado RNG) se reescribe de forma atómica en cada época y se borra al terminar.

`STUDY_DIR/results/` contiene `comparison.md` (tablas listas para el artículo), `summary.csv`, `per_class.csv`, `deltas.csv` y `comparison.json`: media ± DE entre semillas, IC 95 % por bootstrap de fotos para la media entre semillas, IoU por clase y diferencias pareadas de mIoU entre pasos (J0→J1→J2→J3 y cada config vs J0).

Para cambiar algo, crea un YAML nuevo con `base: J2.yaml` y solo los campos que cambian. Un `run_dir` existente rechaza una config distinta, así que no se mezclan resultados.

---

## Añadir otro dataset (p. ej. el de Mauricio): pipeline configurable por esquema de clases

1. Empaquétalo en el mismo formato con `tools/pack_dataset.py` (`images/`, `masks/`, `classes.json`, `manifest.csv` con las columnas de GPS y gimbal, `class_stats.csv`). Súbelo a Drive.
2. Copia `configs/datasets/pascual.yaml` a, por ejemplo, `configs/datasets/mauricio.yaml` y cambia `name`, `version`, `archive` (`filename`, `size_bytes`, `sha256`, `drive_file_id`) y `local.extract_dir`.
3. Esquema de clases, sección `classes:`:
   - `ignore:` clases que se ignoran (p. ej. `[sin_clase]` si significa "sin anotar").
   - Opcional, **mapeo fino → grueso hacia un esquema común**: `target:` (lista ordenada de clases comunes) y `mapping:` (clase fina → clase común). Toda clase no ignorada debe estar mapeada; si falta alguna, el pipeline falla con un mensaje claro. Ejemplo **ilustrativo** (el esquema común hay que acordarlo):

     ```yaml
     classes:
       ignore: [background]
       target: [edificacion, superficie_pavimentada, suelo, vegetacion, agua, vehiculo, otros_objetos]
       mapping:
         Buildings: edificacion
         Road Surface: superficie_pavimentada
         Paved Surface: superficie_pavimentada
         Parkinglots: superficie_pavimentada
         Ground Surface: suelo
         Brownfields: suelo
         Sports Fields: vegetacion
         Vegetation: vegetacion
         Pools: agua
         Vehicle: vehiculo
         Fence: otros_objetos
         Lightning Posts: otros_objetos
     ```

     El YAML de Mauricio llevaría su propio `mapping` (p. ej. `via_pavimentada → superficie_pavimentada`, `vegetacion_alta`/`vegetacion_baja → vegetacion`), con el mismo `target`, para comparar ambos datasets en el esquema común.
4. Usa un `STUDY_DIR` distinto y `prepare_study(STUDY_DIR, "configs/datasets/mauricio.yaml", …)`. Las mismas configs J0–J3/M-b3 sirven: el número de clases sale del esquema.

Hoy el mapeo se aplica a las etiquetas antes de entrenar (el modelo aprende el esquema común). Evaluar en el esquema común un modelo entrenado con el esquema fino (re-mapear predicciones) **no está implementado todavía**.

---

## Pruebas locales (CPU)

```bash
python -m venv .venv && .venv/Scripts/activate          # Windows; en Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest
```

Las pruebas usan datos sintéticos pequeños y cubren: lectura de `classes.json`, RGB → índice (con fallo ante colores desconocidos), mapeo fino → grueso, split por bloques (ninguna foto de train dentro del buffer), métricas contra un caso calculado a mano, bootstrap, clase "no evaluable", descarga por rangos con reanudación (Drive API simulada), validación del pack, reanudación desde checkpoint (misma métrica tras interrumpir), forward/backward de DeepLabv3+ y SegFormer mit-b3, y un flujo de punta a punta estudio → cola → tabla. Para usar el pack real en local: `SEGPACK_ZIP=/ruta/segpack_v1.zip` (se descomprime en `.data/`, ignorado por git).

---

## SAM 2: pre-anotación (`automatic_mask_generator_example.ipynb`)

Genera máscaras zero-shot con `SAM2AutomaticMaskGenerator` sobre las fotos del pack (obtenidas con `ensure_dataset`) y exporta un único `annotations.json` COCO para importarlo en CVAT. Los resultados se escriben en disco local y, opcionalmente, se copian a una carpeta de Drive montado (`RESULTS_DRIVE_DIR`).

## Referencia de comparación

Wilk, Ł. et al. (2022). "Automatic classification of roof types from 3D building models using semantic segmentation." *ISPRS Archives*, XLIII-B2-2022, 485–492. Reportan mIoU 60.3 %, OA 84.1 %, mAcc 81.4 %, mF1 72.0 % (imagen). La comparación es solo orientativa: dataset, clases y protocolo son distintos.

## Licencia

Los componentes de SAM 2 están adaptados de [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) bajo su licencia original.
