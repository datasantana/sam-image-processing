# Corrida de humo en Colab, tiempos y costo

Objetivo: comprobar con GPU que funciona toda la cadena **descarga → validación → entrenamiento → checkpoint → reanudación → métricas**, **medir el tiempo por época** de cada tipo de corrida y **estimar el costo total** de la cola antes de lanzarla.

## Pasos

1. Abre `notebooks/train_ablation.ipynb` en Colab con GPU (anota cuál: los tiempos dependen de ella).
2. Ejecuta las secciones 1–3. La primera vez, la sección 3 pide autorizar tu cuenta de Google (Drive API) y descarga `segpack_v1.zip` (≈560 MB) a `/content/data`. Debe imprimir:
   - `Verificando …segpack_v1.zip (tamaño + sha256)…`
   - `Dataset listo: /content/data/segpack_v1 (66 parejas, origen: drive_api)`
   - el reporte del split: FOV de la cámara, bloques y mezcla de vuelos, y por pliegue train/val/test, fotos excluidas por buffer, distancia mínima train↔test y % de solape residual de huellas (fuga reducida). Además se crea el split de 75 m para J0-b75.
3. Ejecuta la sección 4. Para **J0, J3, J1, J2 y M-b3**, con 6/2/3 fotos, 2 épocas y 32 muestras por época, en `SMOKE_DIR`:
   - primera llamada por config: entrena la época 1, guarda `last.pt` en Drive y se interrumpe a propósito (`→ J0: interrupción simulada tras la época 1`);
   - segunda llamada: `[resume] fold0: continúa en la época 2`, termina, evalúa `best.pt` en val y test, y escribe `DONE.json`;
   - **tabla de tiempos medidos** por config: GPU, micro-lote × acumulación, s de train por muestra, s de val por foto (ventana deslizante a 4032×2268) y s por época del humo;
   - **estimación de costo** de la cola completa (35 corridas con 1 semilla), con los tamaños reales de train/val/test de cada pliegue: s/época estimados por config y horas de GPU entre un mínimo (early stopping lo antes posible: paciencia + 1 épocas) y un máximo (todas las épocas). J0-pasos y J0-b75 se estiman con los tiempos de J0.
4. Comprueba en Drive (`…_smoke/runs/J1/seed42/fold0/`): `config.json`, `split.json`, `env.json` (GPU, versiones, commit, micro-lote), `epoch_log.csv` con 2 filas (incluye `train_seconds`, `val_seconds`, `n_train_samples`, `opt_steps`), `best.pt`, `test/metrics.json`, `test/metrics_per_class.csv`, `test/photo_confusion.npz`, `DONE.json` (con `optimizer_steps_total`). `last.pt` se borra al terminar.
5. Reinicia el entorno y vuelve a ejecutar las secciones 1–4: la sección 4 debe saltar las 5 corridas (`[skip] … ya terminada`) y volver a imprimir tiempos y costo.
6. Revisa el costo. Solo entonces lanza la sección 5. Para probar un corte real, interrumpe la ejecución a mitad de una época y vuelve a ejecutar la celda: debe reanudar desde la última época completa.

Al terminar, borra la carpeta `…_smoke` de Drive si quieres. La estimación lee los `epoch_log.csv` del humo, así que bórrala solo después de estimar.

## Qué se verificó localmente (CPU, sin GPU)

- `pytest`: 40 pruebas, entre ellas:
  - reanudación con métricas, log y pesos idénticos a una corrida sin interrumpir (J0 y J1);
  - geometría de la huella con el FOV real;
  - solape rasterizado;
  - variante de buffer con los mismos val/test;
  - bloques que ignoran la fecha de vuelo;
  - mismo presupuesto de pasos en J0-pasos y J1;
  - humo → estimación de costo, y estudio → cola → tabla con datos sintéticos.
- Con el zip real (copia local, `SEGPACK_ZIP`): sha256, descompresión, validación (66 parejas), split con huellas de cámara, solape residual por pliegue (buffer 50 y 75 m) y mezcla de vuelos por bloque.
- Una versión reducida del humo con las fotos reales en CPU (J1, DeepLabv3+ R18, recortes y ventana de 512): interrupción, reanudación y métricas. Se hizo antes del cambio de huellas; el flujo de entrenamiento no cambió.

## Qué NO se pudo verificar sin Colab

- La descarga real con `google.colab.auth` + Drive API (sí se probó la lógica de rangos y reanudación con un servicio simulado).
- Montaje de Drive y escritura de checkpoints en Drive.
- GPU: AMP (bf16/fp16), tamaños de micro-lote, tiempos reales por época, memoria de la inferencia y de la pérdida de val a 4032×2268. La estimación de costo solo es tan buena como los tiempos del humo.
- La descarga de `nvidia/mit-b3` desde Hugging Face y SegFormer con pesos preentrenados.
- Colab desde la extensión de VS Code.
