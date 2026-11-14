# Corrida de humo en Colab

Objetivo: comprobar en ~10–15 min con GPU que funciona toda la cadena **descarga → validación → entrenamiento → checkpoint → reanudación → métricas** antes de lanzar la cola completa.

## Pasos

1. Abre `notebooks/train_ablation.ipynb` en Colab con GPU.
2. Ejecuta las secciones 1–3. La primera vez, la celda de la sección 3 pide autorizar tu cuenta de Google (Drive API) y descarga `segpack_v1.zip` (≈560 MB) a `/content/data`. Debe imprimir:
   - `Verificando …segpack_v1.zip (tamaño + sha256)…`
   - `Dataset listo: /content/data/segpack_v1 (66 parejas, origen: drive_api)`
   - el reporte del split por pliegue (train/val/test, fotos excluidas por buffer, fotos por clase en val y test).
3. En la sección 4 pon `RUN_SMOKE = True` y ejecútala. Corre J1 (DeepLabv3+) y J3 (SegFormer) con 6/2/3 fotos, 2 épocas y 32 recortes por época, en `STUDY_DIR + "_smoke"`:
   - primera llamada por config: entrena la época 1, guarda `last.pt` en Drive y se interrumpe a propósito (`→ interrupción simulada tras la época 1`);
   - segunda llamada: `[resume] fold0: continúa en la época 2`, termina, evalúa `best.pt` en val y test, y escribe `DONE.json`.
4. Comprueba en Drive (`…_smoke/runs/J1/seed42/fold0/`): `config.json`, `split.json`, `env.json` (GPU, versiones, commit), `epoch_log.csv` con 2 filas, `best.pt`, `test/metrics.json`, `test/metrics_per_class.csv`, `test/photo_confusion.npz`, `DONE.json`. `last.pt` se borra al terminar.
5. Reinicia el entorno (*Entorno de ejecución → Reiniciar*) y vuelve a ejecutar las secciones 1–4. La sección 3 debe decir `Dataset ya presente y validado` si la VM sigue viva, o volver a descargar si es una VM nueva. La sección 4 debe saltar las dos corridas (`[skip] … ya terminada`).
6. Para probar un corte real, lanza la cola (sección 5), interrumpe la ejecución a mitad de una época y vuelve a ejecutar la celda: debe reanudar desde la última época completa.

Al terminar, borra la carpeta `…_smoke` de Drive si quieres.

## Qué se verificó localmente (CPU, sin GPU)

- `pytest`: 34 pruebas, entre ellas reanudación con métricas, log y pesos idénticos a una corrida sin interrumpir (J0 y J1), y un flujo de punta a punta estudio → cola → tabla con datos sintéticos.
- Con el zip real (copia local, `SEGPACK_ZIP`): verificación de sha256, descompresión, validación contra el manifest (66 parejas), split por bloques real y reporte de píxeles ignorados.
- Una versión reducida de esta corrida de humo con las fotos reales en CPU (J1 con DeepLabv3+ R18 ImageNet, recortes y ventana de 512, 8 recortes por época, 3/1/1 fotos): interrupción tras la época 1, `[resume] … continúa en la época 2`, evaluación de `best.pt` y escritura de `DONE.json` y métricas (≈ 4 min en CPU; el mIoU ≈ 0.02 no significa nada con 16 recortes de entrenamiento).

## Qué NO se pudo verificar sin Colab

- La descarga real con `google.colab.auth` + Drive API (sí se probó la lógica de rangos y reanudación con un servicio simulado).
- Montaje de Drive y escritura de checkpoints en Drive.
- GPU: AMP (bf16/fp16), tamaños de micro-lote para crops de 1024, tiempos por época, memoria de la inferencia a 4032×2268.
- La descarga de `nvidia/mit-b3` desde Hugging Face y la corrida de SegFormer con pesos preentrenados.
- Colab desde la extensión de VS Code (si `authenticate_user` falla, `ensure_dataset` pasa a Drive montado; si `drive.mount` tampoco funciona, sube el zip a la VM y usa `SEGPACK_ZIP`).
