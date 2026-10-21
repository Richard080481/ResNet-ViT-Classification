"""Hugging Face ViT construction for full or head-only fine-tuning."""

from __future__ import annotations

from torch import nn

DEFAULT_VIT_CHECKPOINT = "google/vit-base-patch16-224-in21k"


class ViTLogits(nn.Module):
    """Expose Hugging Face image-classification logits as a normal tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def build_vit(
    mode: str,
    num_classes: int = 10,
    checkpoint: str = DEFAULT_VIT_CHECKPOINT,
) -> ViTLogits:
    try:
        from transformers import ViTForImageClassification
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for ViT. Install the project dependencies first."
        ) from exc

    model = ViTForImageClassification.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    if mode == "head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif mode != "full":
        raise ValueError(f"Unknown ViT mode: {mode}")
    return ViTLogits(model)

