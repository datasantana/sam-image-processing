"""Registro de modelos. Todos devuelven logits (N, C, H, W) al tamaño de la entrada."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# mit-b3 (Xie et al., 2021): para construir sin descargar pesos (pruebas en CPU)
SEGFORMER_ARCH = {
    "nvidia/mit-b0": dict(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], decoder_hidden_size=256),
    "nvidia/mit-b3": dict(depths=[3, 4, 18, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768),
}


class SegFormer(nn.Module):
    """SegFormer de transformers con logits reescalados (bilineal) al tamaño de entrada."""

    def __init__(self, hf_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:
            # nvidia/mit-b3 = encoder preentrenado en ImageNet; la cabeza de decodificación se inicializa al azar
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                hf_name, num_labels=num_classes, ignore_mismatched_sizes=True)
        else:
            if hf_name not in SEGFORMER_ARCH:
                raise ValueError(f"Sin arquitectura local para {hf_name}; usa pretrained=True")
            cfg = SegformerConfig(num_labels=num_classes, **SEGFORMER_ARCH[hf_name])
            self.net = SegformerForSemanticSegmentation(cfg)

    def forward(self, x):
        logits = self.net(pixel_values=x).logits  # (N, C, H/4, W/4)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(model_cfg: dict, num_classes: int) -> nn.Module:
    arch = model_cfg["arch"].lower()
    weights = model_cfg.get("weights")
    pretrained = weights not in (None, "none", False)
    if arch == "segformer":
        return SegFormer(model_cfg.get("hf_name") or "nvidia/mit-b3", num_classes, pretrained=pretrained)
    import segmentation_models_pytorch as smp
    registry = {
        "deeplabv3plus": smp.DeepLabV3Plus,
        "pspnet": smp.PSPNet,
        "unetplusplus": smp.UnetPlusPlus,
    }
    if arch not in registry:
        raise ValueError(f"Modelo desconocido '{arch}'. Opciones: {sorted(registry) + ['segformer']}")
    return registry[arch](encoder_name=model_cfg.get("encoder", "resnet50"),
                          encoder_weights=weights if pretrained else None,
                          in_channels=3, classes=num_classes)


def input_multiple(model_cfg: dict) -> int:
    """Múltiplo requerido del tamaño de entrada (para el relleno en inferencia)."""
    arch = model_cfg["arch"].lower()
    return {"deeplabv3plus": 16, "pspnet": 8, "unetplusplus": 32, "segformer": 32}.get(arch, 32)
