"""
visualization.py — GradCAM overlay and chart helpers.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io


def overlay_gradcam(original_image: Image.Image,
                    cam: np.ndarray,
                    alpha: float = 0.45) -> Image.Image:
    """Overlay GradCAM heatmap on original leaf image."""
    img_resized = original_image.resize((224, 224)).convert("RGB")
    img_array   = np.array(img_resized, dtype=np.float32) / 255.0

    colormap    = cm.get_cmap("RdYlGn_r")
    heatmap_rgb = colormap(cam)[:, :, :3]

    blended = (1 - alpha) * img_array + alpha * heatmap_rgb
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def make_confidence_bar_chart(top_k_predictions: list) -> bytes:
    """Create horizontal bar chart of top-k predictions."""
    labels = [_fmt(cls) for cls, _ in top_k_predictions]
    values = [conf for _, conf in top_k_predictions]
    colors = ["#2d6a4f" if i == 0 else "#a8dadc"
              for i in range(len(values))]

    fig, ax = plt.subplots(
        figsize=(6, max(2, len(labels) * 0.65)))
    bars = ax.barh(labels[::-1], values[::-1],
                   color=colors[::-1], height=0.55)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center",
                ha="left", fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _fmt(class_name: str) -> str:
    """'Tomato___Early_blight' → 'Tomato: Early blight'"""
    parts  = class_name.split("___")
    plant  = parts[0].replace("_", " ")
    if len(parts) > 1:
        disease = parts[1].replace("_", " ").capitalize()
        return f"{plant}: {disease}"
    return plant


def severity_color(severity: str) -> str:
    return {
        "None":     "#2a9d8f",
        "Low":      "#e9c46a",
        "Moderate": "#f4a261",
        "Severe":   "#e76f51",
        "Unknown":  "#aaaaaa",
    }.get(severity, "#aaaaaa")