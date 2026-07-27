#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraînement du CNN de localisation (TinyDroneLocalizer) sur le dataset ROI généré par
generate_dataset.py (Dataset_CNN/train + Dataset_CNN/val).

Principaux changements par rapport à la V2 :
- Sauvegarde de checkpoints (meilleur modèle + dernier modèle) : la V2 n'en sauvegardait
  aucun, ce qui rendait evaluate_cnn.py / export_onnx.py incapables de charger un vrai
  modèle entraîné (ils cherchaient des fichiers .pth jamais créés).
- Early stopping + réduction du taux d'apprentissage sur plateau : sur la dernière run,
  la loss de validation était minimale vers l'époque 7 (~0.127) puis remontait jusqu'à
  0.140 à l'époque 20 alors que la loss d'entraînement continuait de baisser -> surapprentissage
  net. Le modèle exporté en ONNX était donc la pire version des 20, pas la meilleure.
- Architecture allégée : Global Average Pooling à la place de l'aplatissement brut
  (128*8*8 = 8192 -> 256, ~2.1M paramètres rien que sur cette couche), qui est la source
  la plus probable du surapprentissage vu la taille du dataset.
- Métriques affichées en unités physiques (mètres, degrés) en plus de la loss brute,
  pour pouvoir juger la précision réelle du modèle d'un coup d'œil.
- L'export ONNX se fait à partir du MEILLEUR modèle (rechargé depuis le checkpoint),
  jamais du dernier.
"""

import os
import glob
import json
import math
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "train_dir": "./Dataset_CNN/train",
    "val_dir": "./Dataset_CNN/val",
    "checkpoint_dir": "./checkpoints",
    "onnx_path": "tiny_drone_localizer.onnx",

    "batch_size": 32,
    "max_epochs": 60,          # Plafond haut : l'early stopping décide de l'arrêt réel
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "grad_clip_norm": 5.0,

    "early_stopping_patience": 8,   # Nb d'époques sans amélioration de la val_loss avant arrêt
    "lr_scheduler_patience": 3,     # Réduit le LR si la val_loss stagne 3 époques
    "lr_scheduler_factor": 0.5,

    "seed": 42,

    # Normalisation des labels -> DOIT rester cohérente avec generate_dataset.py (CONFIG)
    "pos_xy_range_m": 6.0,       # x,y normalisés par /6.0
    "altitude_range_m": (2.0, 6.0),
    "roll_max_deg": 35.0,
    "pitch_max_deg": 20.0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. DATASET : conversion du yaw en sin/cos + légère augmentation photométrique
# ==============================================================================
class DronePoseDataset(Dataset):
    """
    Charge les crops ROI (image + JSON target_pose_xyz_rpy) générés par generate_dataset.py.
    train=True active une légère augmentation photométrique (luminosité/contraste/bruit) :
    aucune augmentation géométrique (flip/rotation) n'est appliquée car elle inverserait le
    signe de roll/yaw sans certitude de cohérence avec le reste du pipeline.
    """

    def __init__(self, data_dir, config, train=False):
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"Aucune image trouvée dans {data_dir} !")
        self.config = config
        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def _augment(self, img):
        alpha = random.uniform(0.85, 1.15)   # contraste
        beta = random.uniform(-15, 15)       # luminosité
        img = img.astype(np.float32) * alpha + beta
        if random.random() < 0.5:
            img += np.random.normal(0, 4.0, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        json_path = img_path.replace(".png", ".json")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.train:
            img = self._augment(img)

        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5

        with open(json_path, 'r') as f:
            data = json.load(f)
        labels = data["target_pose_xyz_rpy"]  # [X, Y, Z, Roll, Pitch, Yaw]

        cfg = self.config
        x_norm = labels[0] / cfg["pos_xy_range_m"]
        y_norm = labels[1] / cfg["pos_xy_range_m"]
        alt_min, alt_max = cfg["altitude_range_m"]
        z_norm = (labels[2] - alt_min) / (alt_max - alt_min)

        roll_norm = labels[3] / cfg["roll_max_deg"]
        pitch_norm = labels[4] / cfg["pitch_max_deg"]

        yaw_rad = math.radians(labels[5])
        sin_yaw = math.sin(yaw_rad)
        cos_yaw = math.cos(yaw_rad)

        normalized_labels = [x_norm, y_norm, z_norm, roll_norm, pitch_norm, sin_yaw, cos_yaw]
        return img_tensor, torch.tensor(normalized_labels, dtype=torch.float32)


# ==============================================================================
# 2. ARCHITECTURE : Global Average Pooling au lieu d'un flatten brut
# ==============================================================================
class TinyDroneLocalizer(nn.Module):
    """
    Même profondeur que la V2 (4 blocs conv, downsampling x2 à chaque étage) mais la tête
    de régression passe par un Global Average Pooling plutôt qu'un flatten complet :
    128*8*8=8192 -> 256 (≈2.1M paramètres) devient 128 -> 128 -> 7 (≈17k paramètres).
    C'est la source la plus probable du surapprentissage observé avec un dataset de
    quelques dizaines de milliers d'images.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 128 -> 64
            nn.BatchNorm2d(32), nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(64), nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(128), nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, 128, 8, 8) -> (B, 128, 1, 1)
        self.regressor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7),  # x, y, z, roll, pitch, sin(yaw), cos(yaw)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.regressor(x)


# ==============================================================================
# 3. MÉTRIQUES EN UNITÉS PHYSIQUES (pour un suivi lisible, pas juste une loss abstraite)
# ==============================================================================
def denormalize_batch(labels, cfg):
    """labels: tenseur (B,7) normalisé -> dict de tenseurs en unités physiques (m, deg)."""
    alt_min, alt_max = cfg["altitude_range_m"]
    x_m = labels[:, 0] * cfg["pos_xy_range_m"]
    y_m = labels[:, 1] * cfg["pos_xy_range_m"]
    z_m = labels[:, 2] * (alt_max - alt_min) + alt_min
    roll_deg = labels[:, 3] * cfg["roll_max_deg"]
    pitch_deg = labels[:, 4] * cfg["pitch_max_deg"]
    yaw_deg = torch.atan2(labels[:, 5], labels[:, 6]) * (180.0 / math.pi)
    return {"x": x_m, "y": y_m, "z": z_m, "roll": roll_deg, "pitch": pitch_deg, "yaw": yaw_deg}


def compute_physical_mae(outputs, targets, cfg):
    """Erreur absolue moyenne en unités physiques, pour juger la précision réelle du modèle."""
    pred = denormalize_batch(outputs.detach(), cfg)
    true = denormalize_batch(targets.detach(), cfg)

    pos_error_m = torch.sqrt(
        (pred["x"] - true["x"]) ** 2 + (pred["y"] - true["y"]) ** 2 + (pred["z"] - true["z"]) ** 2
    ).mean().item()
    roll_error_deg = (pred["roll"] - true["roll"]).abs().mean().item()
    pitch_error_deg = (pred["pitch"] - true["pitch"]).abs().mean().item()

    yaw_diff = (pred["yaw"] - true["yaw"] + 180.0) % 360.0 - 180.0  # écart circulaire correct
    yaw_error_deg = yaw_diff.abs().mean().item()

    return pos_error_m, roll_error_deg, pitch_error_deg, yaw_error_deg


# ==============================================================================
# 4. ENTRAÎNEMENT
# ==============================================================================
def run_epoch(model, loader, criterion, cfg, optimizer=None):
    """Une passe (train si optimizer fourni, sinon eval). Retourne loss + MAE physiques moyennes."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_pos, total_roll, total_pitch, total_yaw = 0.0, 0.0, 0.0, 0.0
    n_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            batch_size = images.size(0)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
                optimizer.step()

            pos_err, roll_err, pitch_err, yaw_err = compute_physical_mae(outputs, targets, cfg)

            total_loss += loss.item() * batch_size
            total_pos += pos_err * batch_size
            total_roll += roll_err * batch_size
            total_pitch += pitch_err * batch_size
            total_yaw += yaw_err * batch_size
            n_samples += batch_size

    return {
        "loss": total_loss / n_samples,
        "pos_m": total_pos / n_samples,
        "roll_deg": total_roll / n_samples,
        "pitch_deg": total_pitch / n_samples,
        "yaw_deg": total_yaw / n_samples,
    }


def train(config=CONFIG):
    set_seed(config["seed"])
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    print(f"🚀 Préparation de l'entraînement (device: {DEVICE})...")
    train_dataset = DronePoseDataset(config["train_dir"], config, train=True)
    val_dataset = DronePoseDataset(config["val_dir"], config, train=False)
    print(f"   {len(train_dataset)} images train / {len(val_dataset)} images val")

    persistent = config["num_workers"] > 0
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=persistent,
    )

    model = TinyDroneLocalizer().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Modèle : {n_params:,} paramètres")

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config["lr_scheduler_factor"], patience=config["lr_scheduler_patience"]
    )
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path = os.path.join(config["checkpoint_dir"], "tiny_drone_best.pth")
    last_checkpoint_path = os.path.join(config["checkpoint_dir"], "tiny_drone_last.pth")

    print(f"📈 Début de l'apprentissage (plafond {config['max_epochs']} époques, "
          f"early stopping patience={config['early_stopping_patience']})...\n")

    for epoch in range(1, config["max_epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, config, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, config, optimizer=None)

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{config['max_epochs']}] "
            f"Loss train={train_metrics['loss']:.5f} val={val_metrics['loss']:.5f} | "
            f"MAE val -> pos={val_metrics['pos_m']*100:.1f}cm "
            f"roll={val_metrics['roll_deg']:.2f}° pitch={val_metrics['pitch_deg']:.2f}° "
            f"yaw={val_metrics['yaw_deg']:.2f}° | lr={current_lr:.2e}"
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "config": config,
        }, last_checkpoint_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_metrics["loss"],
                "config": config,
            }, best_checkpoint_path)
            print(f"   ✅ Nouveau meilleur modèle (val_loss={best_val_loss:.5f}) -> {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["early_stopping_patience"]:
                print(f"\n⏹️ Early stopping : pas d'amélioration depuis "
                      f"{config['early_stopping_patience']} époques (meilleure val_loss={best_val_loss:.5f}).")
                break

    # --- Export ONNX à partir du MEILLEUR modèle, jamais du dernier ---
    print(f"\n📦 Rechargement du meilleur modèle ({best_checkpoint_path}) pour l'export ONNX...")
    best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval().cpu()

    dummy_input = torch.randn(1, 3, 128, 128)
    torch.onnx.export(
        model, dummy_input, config["onnx_path"],
        export_params=True, opset_version=12,
        input_names=["input_roi"], output_names=["output_pose"],
    )
    print(f"✨ Modèle ONNX généré : {config['onnx_path']} "
          f"(depuis l'époque {best_checkpoint['epoch']}, val_loss={best_checkpoint['val_loss']:.5f})")


if __name__ == "__main__":
    train()
