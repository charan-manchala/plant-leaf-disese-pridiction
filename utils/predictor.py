"""
predictor.py — Inference pipeline for plant leaf disease prediction.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

IMG_SIZE = 224

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Model builder ─────────────────────────────────────────────────────────────
def build_inference_model(num_classes: int,
                          model_name: str = "efficientnet_b0"):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model


# ── GradCAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    """Lightweight GradCAM for disease region highlighting."""

    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam


def estimate_disease_percentage(cam: np.ndarray,
                                threshold: float = 0.4) -> float:
    """Estimate % of leaf affected using GradCAM heatmap."""
    affected = (cam > threshold).sum()
    return round((affected / cam.size) * 100, 1)


# ── Main Predictor ────────────────────────────────────────────────────────────
class PlantDiseasePredictor:
    """
    Full pipeline:
      1. Preprocess image
      2. Forward pass → top-k predictions
      3. GradCAM heatmap
      4. Disease percentage estimate
      5. Return structured result dict
    """

    def __init__(self,
                 model_path: str,
                 class_indices_path: str,
                 model_name: str = "efficientnet_b0",
                 device: str = None):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load class index mapping
        with open(class_indices_path, "r") as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {int(v): k
                             for k, v in class_to_idx.items()}
        num_classes = len(self.idx_to_class)

        # Load model weights
        self.model = build_inference_model(num_classes, model_name)
        state = torch.load(model_path,
                           map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # GradCAM setup
        if model_name == "efficientnet_b0":
            target_layer = self.model.features[-1]
        else:
            target_layer = self.model.layer4[-1]
        self.gradcam = GradCAM(self.model, target_layer)

        print(f"Model loaded  : {model_name}")
        print(f"Classes       : {num_classes}")
        print(f"Device        : {self.device}")

    def _parse_label(self, label: str):
        """
        'Tomato___Early_blight' →
        plant='Tomato', is_healthy=False, disease='Early blight'
        """
        parts   = label.split("___")
        plant   = parts[0].replace("_", " ").strip()
        if len(parts) == 1:
            return plant, True, "Healthy"
        disease    = parts[1].replace("_", " ").strip()
        is_healthy = "healthy" in disease.lower()
        return plant, is_healthy, disease

    def predict(self, pil_image: Image.Image,
                top_k: int = 3) -> dict:
        """
        Run full prediction on a PIL image.

        Returns dict:
          plant_name, plant_type, is_healthy,
          disease_name, disease_key,
          confidence, disease_percentage,
          top_k_predictions, cam_array
        """
        from data.disease_cure_db import PLANT_TYPES

        # Preprocess
        tensor = INFER_TRANSFORM(pil_image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1)[0]

        # Top-k
        top_probs, top_indices = torch.topk(
            probs, k=min(top_k, len(self.idx_to_class)))
        top_k_preds = [
            (self.idx_to_class[idx.item()],
             round(p.item() * 100, 2))
            for idx, p in zip(top_indices, top_probs)
        ]

        best_idx   = top_indices[0].item()
        best_class = self.idx_to_class[best_idx]
        confidence = round(top_probs[0].item() * 100, 2)

        plant_name, is_healthy, disease_name = \
            self._parse_label(best_class)

        # GradCAM for disease %
        cam_array          = None
        disease_percentage = None
        if not is_healthy:
            t = INFER_TRANSFORM(pil_image).unsqueeze(0).to(
                self.device).requires_grad_(True)
            cam_array = self.gradcam.generate(t, best_idx)
            disease_percentage = estimate_disease_percentage(cam_array)

        plant_key  = best_class.split("___")[0]
        plant_type = PLANT_TYPES.get(plant_key, "Crop plant")

        return {
            "plant_name":        plant_name,
            "plant_type":        plant_type,
            "is_healthy":        is_healthy,
            "disease_name":      disease_name,
            "disease_key":       best_class,
            "confidence":        confidence,
            "disease_percentage": disease_percentage,
            "top_k_predictions": top_k_preds,
            "cam_array":         cam_array,
        }