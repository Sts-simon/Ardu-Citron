#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_dataset_diversity.py

Analyse la diversité d'un dataset produit par generate_dataset.py et génère un
tableau de bord (PNG) de graphiques, plus un résumé texte dans le terminal.

Fonctionne avec les deux types de dataset (détection automatique du format) :
  - Dataset_CNN/train + Dataset_CNN/val   -> exemples indépendants (pose only)
  - Dataset_Verification/                  -> trajectoires (pose + environnement + IMU)

Usage:
    python3 check_dataset_diversity.py Dataset_CNN
    python3 check_dataset_diversity.py Dataset_Verification
    python3 check_dataset_diversity.py Dataset_CNN Dataset_Verification
    python3 check_dataset_diversity.py                # essaie les deux noms par défaut
    python3 check_dataset_diversity.py --sample-images 800 Dataset_CNN
"""

import os
import sys
import glob
import json
import random
import argparse
import math
from collections import Counter

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# UTILITAIRES
# ==============================================================================

def load_jsons(directory):
    """Charge tous les .json d'un dossier (non récursif). Retourne (data, json_paths)."""
    paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    data, kept_paths = [], []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data.append(json.load(f))
            kept_paths.append(p)
        except Exception:
            continue
    return data, kept_paths


def normalized_entropy(counter):
    """Entropie de Shannon normalisée [0,1] : 1 = parfaitement équilibré, 0 = un seul type."""
    total = sum(counter.values())
    if total == 0 or len(counter) <= 1:
        return 1.0
    probs = np.array([c / total for c in counter.values()])
    ent = -np.sum(probs * np.log(probs))
    return float(ent / math.log(len(counter)))


def sample_image_stats(png_paths, n_sample):
    """Échantillonne des images pour mesurer la diversité photométrique réelle des pixels
    (luminosité moyenne, saturation moyenne) — utile car le JSON du dataset CNN ne stocke
    pas les paramètres d'environnement (texture, éclairage) contrairement au dataset de
    vérification."""
    if not png_paths:
        return np.array([]), np.array([])
    sample = random.sample(png_paths, min(n_sample, len(png_paths)))
    brightness, saturation = [], []
    for p in sample:
        img = cv2.imread(p)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness.append(hsv[:, :, 2].mean())
        saturation.append(hsv[:, :, 1].mean())
    return np.array(brightness), np.array(saturation)


def hist_panel(ax, values, title, bins=30, color="#4C72B0"):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        ax.set_title(f"{title} (no data)")
        return
    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.3)
    ax.set_title(f"{title}\nmean={values.mean():.2f}  std={values.std():.2f}  "
                 f"range=[{values.min():.2f}, {values.max():.2f}]", fontsize=9)
    ax.tick_params(labelsize=8)


def bar_panel(ax, counter, title, rotate=30):
    if not counter:
        ax.set_title(f"{title} (no data)")
        return
    items = sorted(counter.items(), key=lambda kv: str(kv[0]))
    labels = [str(k) for k, _ in items]
    values = [v for _, v in items]
    ax.bar(labels, values, color="#55A868", edgecolor="white", linewidth=0.3)
    ent = normalized_entropy(counter)
    ax.set_title(f"{title}  (balance={ent:.2f})", fontsize=9)
    ax.tick_params(axis="x", labelrotation=rotate, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)


def scatter_panel(ax, x, y, title, xlabel="X (m)", ylabel="Y (m)"):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) == 0:
        ax.set_title(f"{title} (no data)")
        return
    ax.hist2d(x, y, bins=40, cmap="viridis")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def print_section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ==============================================================================
# DATASET CNN (Dataset_CNN/train + Dataset_CNN/val) : exemples indépendants
# ==============================================================================
def analyze_cnn_dataset(root, sample_images, out_path):
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    train_data, train_json_paths = load_jsons(train_dir) if os.path.isdir(train_dir) else ([], [])
    val_data, val_json_paths = load_jsons(val_dir) if os.path.isdir(val_dir) else ([], [])
    all_data = train_data + val_data
    all_json_paths = train_json_paths + val_json_paths

    if not all_data:
        print(f"❌ No JSON found under '{root}/train' or '{root}/val'. Skipping.")
        return

    poses = np.array([d["target_pose_xyz_rpy"] for d in all_data], dtype=float)
    x, y, z, roll, pitch, yaw = poses.T
    marker_ids = Counter(d.get("marker_id") for d in all_data)
    splits = Counter(d.get("split", "?") for d in all_data)

    png_paths = [p.replace(".json", ".png") for p in all_json_paths]
    brightness, saturation = sample_image_stats(png_paths, sample_images)

    print_section(f"CNN DATASET — {root}")
    print(f"Total examples      : {len(all_data)}  (train={len(train_data)}, val={len(val_data)})")
    print(f"Markers represented  : {len(marker_ids)}  -> balance score {normalized_entropy(marker_ids):.2f} (1.0 = perfectly even)")
    print(f"Train/val balance    : {dict(splits)}")
    print(f"X range   : [{x.min():.2f}, {x.max():.2f}] m   std={x.std():.2f}")
    print(f"Y range   : [{y.min():.2f}, {y.max():.2f}] m   std={y.std():.2f}")
    print(f"Z range   : [{z.min():.2f}, {z.max():.2f}] m   std={z.std():.2f}")
    print(f"Roll/Pitch/Yaw std : {roll.std():.2f}° / {pitch.std():.2f}° / {yaw.std():.2f}°")
    if len(brightness):
        print(f"Image brightness (sample n={len(brightness)}): mean={brightness.mean():.1f} std={brightness.std():.1f}")
        print(f"Image saturation (sample n={len(saturation)}): mean={saturation.mean():.1f} std={saturation.std():.1f}")

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(f"CNN training dataset diversity — {root}  (n={len(all_data)})", fontsize=13, fontweight="bold")

    hist_panel(axes[0, 0], x, "X position (m)")
    hist_panel(axes[0, 1], y, "Y position (m)")
    hist_panel(axes[0, 2], z, "Altitude Z (m)")
    hist_panel(axes[1, 0], roll, "Roll (deg)")
    hist_panel(axes[1, 1], pitch, "Pitch (deg)")
    hist_panel(axes[1, 2], yaw, "Yaw (deg)")
    scatter_panel(axes[2, 0], x, y, "Marker position coverage (X vs Y)")
    bar_panel(axes[2, 1], marker_ids, "Examples per marker ID")
    if len(brightness):
        hist_panel(axes[2, 2], brightness, "Image brightness (sampled)", color="#C44E52")
    else:
        axes[2, 2].set_title("Image brightness (no images found)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"📊 Saved dashboard: {out_path}")


# ==============================================================================
# DATASET DE VÉRIFICATION (trajectoires continues) : pose + environnement + IMU
# ==============================================================================
def analyze_verification_dataset(root, sample_images, out_path):
    data, json_paths = load_jsons(root)
    if not data:
        print(f"❌ No JSON found under '{root}'. Skipping.")
        return

    distance = np.array([d["distance_m"] for d in data], dtype=float)
    roll = np.array([d["roll_deg"] for d in data], dtype=float)
    pitch = np.array([d["pitch_deg"] for d in data], dtype=float)
    yaw = np.array([d["yaw_deg"] for d in data], dtype=float)
    marker_px = np.array([d["marker_pixels"] for d in data], dtype=float)
    fully_in_frame = Counter(bool(d.get("marker_fully_in_frame", False)) for d in data)
    drone_x = np.array([d["drone_pos_m"][0] for d in data], dtype=float)
    drone_y = np.array([d["drone_pos_m"][1] for d in data], dtype=float)
    marker_ids = Counter(d.get("marker_id") for d in data)
    trajectories = Counter(d.get("trajectory_id") for d in data)
    textures = Counter(d.get("environment", {}).get("ground_texture", "?") for d in data)
    ae_gain = np.array([d.get("environment", {}).get("ae_gain", np.nan) for d in data], dtype=float)
    vignette = np.array([d.get("environment", {}).get("vignette_strength", np.nan) for d in data], dtype=float)
    wb_temp = np.array([d.get("environment", {}).get("wb_temp", np.nan) for d in data], dtype=float)
    vibration = np.array([d.get("environment", {}).get("vibration_px", np.nan) for d in data], dtype=float)
    frames_per_traj = np.array(list(trajectories.values()), dtype=float)

    print_section(f"VERIFICATION DATASET — {root}")
    print(f"Total frames         : {len(data)}")
    print(f"Trajectories         : {len(trajectories)}  (frames/trajectory: "
          f"min={frames_per_traj.min():.0f} max={frames_per_traj.max():.0f} mean={frames_per_traj.mean():.1f})")
    print(f"Markers represented  : {len(marker_ids)}  -> balance score {normalized_entropy(marker_ids):.2f}")
    print(f"Ground textures      : {dict(textures)}  -> balance score {normalized_entropy(textures):.2f}")
    print(f"Marker fully in frame: {dict(fully_in_frame)}")
    print(f"Marker size (px)     : range=[{marker_px.min():.0f}, {marker_px.max():.0f}] mean={marker_px.mean():.0f} std={marker_px.std():.0f}")
    print(f"Roll/Pitch/Yaw std   : {roll.std():.2f}° / {pitch.std():.2f}° / {yaw.std():.2f}°")

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    fig.suptitle(f"Verification dataset diversity — {root}  (n={len(data)})", fontsize=13, fontweight="bold")

    hist_panel(axes[0, 0], distance, "Altitude / distance (m)")
    hist_panel(axes[0, 1], roll, "Roll (deg)")
    hist_panel(axes[0, 2], pitch, "Pitch (deg)")
    hist_panel(axes[1, 0], yaw, "Yaw (deg)")
    scatter_panel(axes[1, 1], drone_x, drone_y, "Drone position coverage (X vs Y)")
    hist_panel(axes[1, 2], marker_px, "Marker size in frame (px)", color="#C44E52")

    bar_panel(axes[2, 0], marker_ids, "Frames per marker ID")
    bar_panel(axes[2, 1], textures, "Ground texture distribution")
    bar_panel(axes[2, 2], Counter({str(k): v for k, v in fully_in_frame.items()}), "Marker fully in frame?")

    hist_panel(axes[3, 0], ae_gain, "Auto-exposure gain", color="#8172B2")
    hist_panel(axes[3, 1], vignette, "Vignette strength", color="#8172B2")
    hist_panel(axes[3, 2], frames_per_traj, "Frames per trajectory (early exit spread)", color="#CCB974")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"📊 Saved dashboard: {out_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def detect_dataset_type(root):
    """Retourne 'cnn' si train/ ou val/ existent avec des .json, 'verification' si des .json
    à la racine ont un champ 'trajectory_id', sinon None."""
    if os.path.isdir(os.path.join(root, "train")) or os.path.isdir(os.path.join(root, "val")):
        return "cnn"
    sample = glob.glob(os.path.join(root, "*.json"))[:1]
    if sample:
        try:
            with open(sample[0]) as f:
                d = json.load(f)
            if "trajectory_id" in d:
                return "verification"
            if "target_pose_xyz_rpy" in d:
                return "cnn"
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Check dataset diversity (generate_dataset.py output)")
    parser.add_argument("datasets", nargs="*", default=["Dataset_CNN", "Dataset_Verification"],
                         help="Dataset folder(s) to analyze (default: Dataset_CNN Dataset_Verification)")
    parser.add_argument("--sample-images", type=int, default=500,
                         help="Number of images to sample for photometric diversity checks (default: 500)")
    parser.add_argument("--out-dir", default=".",
                         help="Where to save the dashboard PNGs (default: current directory)")
    args = parser.parse_args()

    random.seed(0)
    found_any = False

    for root in args.datasets:
        if not os.path.isdir(root):
            print(f"⏭️  '{root}' not found, skipping.")
            continue

        dtype = detect_dataset_type(root)
        safe_name = root.strip("./").replace(os.sep, "_")
        out_path = os.path.join(args.out_dir, f"diversity_{safe_name}.png")

        if dtype == "cnn":
            analyze_cnn_dataset(root, args.sample_images, out_path)
            found_any = True
        elif dtype == "verification":
            analyze_verification_dataset(root, args.sample_images, out_path)
            found_any = True
        else:
            print(f"⏭️  Could not recognize the dataset format in '{root}' (no matching JSON found).")

    if not found_any:
        print("\nNo recognizable dataset found. Pass a folder explicitly, e.g.:\n"
              "  python3 check_dataset_diversity.py Dataset_CNN\n"
              "  python3 check_dataset_diversity.py Dataset_Verification")


if __name__ == "__main__":
    main()
