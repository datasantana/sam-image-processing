# M-b3: reproducción aproximada de la receta de Mauricio

Fuente: `Metricas_run05_b3_v2.xlsx` (hojas `LEEME`, `Config`, `IoU_por_clase`, `Matriz_confusion`, `Log_epocas`), solo lectura. **No tenemos su script de entrenamiento**, así que `configs/runs/M-b3.yaml` es una reproducción aproximada: entrena *su receta* sobre *nuestro* dataset y protocolo, para que la comparación con J0–J3 sea justa. No reproduce sus números, que vienen de otro dataset (v2, 10 clases, otra zona).

## Tomado literalmente de la hoja `Config`

| Parámetro (hoja) | Valor | En `M-b3.yaml` |
|---|---|---|
| modelo | `nvidia/mit-b3` | `model: {arch: segformer, hf_name: nvidia/mit-b3}` |
| crop | 1024 | `data.crop: 1024` |
| esc_entr | [0.5, 1.0] | `data.scale_range: [0.5, 1.0]` (ver interpretación abajo) |
| esc_inf | 0.75 | `infer.scale: 0.75` |
| lote | 8 | `train.batch: 8` (efectivo; con acumulación si la GPU no da) |
| lr | 6e-05 | `optim.lr: 6.0e-5` |
| epocas_max | 100 | `train.epochs: 100` |
| muestras_epoca | 880 | `data.samples_per_epoch: 880` (110 pasos de 8) |
| paciencia | 15 | `train.patience: 15` |
| semilla | 42 | nuestra semilla por defecto (42; extra con `EXTRA_SEEDS`) |
| pesos_clase | None | `loss.weights: none` |
| torch / transformers | 2.11.0+cu130 / 5.20.2 | se usa lo que trae Colab; las versiones quedan en `env.json` |

## Tomado de la hoja `LEEME`

- Selección del checkpoint "mejor max-min" (época 32 de 47): se interpreta como **maximizar el IoU mínimo por clase en validación** → `train.select: maxmin`. El `Log_epocas` (columnas `min_criterio`, `media_criterio`) concuerda con esa lectura.
- Su `sin_clase` se entrenaba como clase y solo se excluía del criterio. En nuestro protocolo, *background* (sin anotar) se **ignora** en pérdida, métricas y matriz de confusión en todos los pasos, M-b3 incluido. Esta diferencia es deliberada.

## Implementado, con interpretación propia (SUPUESTO)

| Aspecto | Cómo está implementado | Dónde |
|---|---|---|
| **(a) esc_entr [0.5, 1.0]** | Sí. Por cada muestra se sortea s ~ **Uniforme continua [0.5, 1.0]**, se recorta una ventana de 1024/s px nativos (1024–2048 px) y se reescala a 1024×1024 (bilineal la imagen, vecino más cercano la máscara). Equivale a reescalar la foto por s y recortar 1024. **Supuesto:** distribución uniforme continua y reescalado por muestra (no por foto ni por época). | `segpipe/data.py::CropDataset` |
| **(b) mejor_maxmin** | Sí. En cada época se calcula el IoU por clase en val (matriz de confusión acumulada a resolución nativa) y se guarda `best.pt` cuando **mejora el mínimo** de esos IoU. La paciencia (15) cuenta épocas sin mejorar ese mínimo. **Supuesto:** el mínimo se toma sobre las clases presentes en ≥ 1 foto de val (`select_min_photos: 1`), sin background; Mauricio excluía `sin_clase` del criterio. | `segpipe/train.py` (`select: maxmin`) |
| esc_inf 0.75 | Foto completa reescalada a 0.75; ventana deslizante sobre esa foto; logits devueltos a la resolución nativa (bilineal) antes del argmax. | `segpipe/infer.py` |

## Supuesto (no está en la hoja; valores típicos de SegFormer)

| Aspecto | Supuesto | Por qué |
|---|---|---|
| Optimizador | AdamW, weight decay 0.01 | lr 6e-5 es el valor estándar de SegFormer con AdamW (mmseg y tutoriales HF); 0.01 es el weight decay por defecto de `torch.optim.AdamW` |
| Scheduler | poly (potencia 1.0) por iteración, sin warmup | el más común con SegFormer; sin datos sobre warmup |
| Pérdida | cross-entropy sin pesos | `pesos_clase = None`; SegFormer de HF usa CE |
| Aumentos | reescalado uniforme en [0.5, 1.0] + recorte 1024 + flip horizontal | la hoja solo da la escala; el flip es el aumento mínimo habitual |
| Inferencia | foto reescalada a 0.75 + ventana deslizante 1024 con 50 % de solape (promedio de logits) | la hoja solo da la escala; ventana = crop de entrenamiento |
| Normalización | media/DE de ImageNet | es la de `SegformerImageProcessor` |
| Muestreo de recortes | uniforme (foto y posición) | no se menciona sesgo a clases |

Si consigues el script de Mauricio, corrige estos campos en `configs/runs/M-b3.yaml`. Como `train_run` rechaza un `run_dir` cuya config no coincide, las corridas viejas no se mezclan con las nuevas. Para conservar ambas, renombra el archivo (p. ej. `M-b3v2.yaml`).
