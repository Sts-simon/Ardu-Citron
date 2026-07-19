import os
import subprocess
import json
import numpy as np
import sys
import matplotlib.pyplot as plt

DATASET_DIR = "/mnt/1to/ODB/Ardu-Citron-3/Sol/Markers_5/Dataset"
RUST_BIN = "./drone_localizer/target/release/drone_localizer"
TOTAL_IMAGES = 7500  

print(f"🚀 Lancement du benchmark visuel (Kalman Vision + IMU)...")

process = subprocess.Popen(
    [RUST_BIN, DATASET_DIR, str(TOTAL_IMAGES)],
    stdout=subprocess.PIPE,
    text=True
)

frames = []
true_dists, est_dists = [], []
true_rolls, est_rolls = [], []
true_pitchs, est_pitchs = [], []
true_yaws, est_yaws = [], []
execution_times = []

count = 0

for line in process.stdout:
    line = line.strip()
    if line.startswith("DATA|") or line.startswith("NODATA|"):
        count += 1
        
        bar_length = 30
        progress = min(count / TOTAL_IMAGES, 1.0)
        block = int(round(bar_length * progress))
        sys.stdout.write(f"\r⌛ Progression: [{'█' * block + '-' * (bar_length - block)}] {count}/{TOTAL_IMAGES} images")
        sys.stdout.flush()
        
        if line.startswith("DATA|"):
            parts = line.split("|")
            file_name = parts[1]
            
            # ALIGNEMENT STRICT DES INDEX AVEC LE PRINTLN! RUST
            est_dist = float(parts[2])   # kalman.distance
            # parts[3] = tz brut (non utilisé ici, gardé pour debug côté Rust)
            est_roll = float(parts[4])   # kalman.roll
            est_pitch = float(parts[5])  # kalman.pitch
            est_yaw = float(parts[6])    # kalman.yaw
            exec_ms = float(parts[7])    # duration_ms
            
            json_name = file_name.replace(".png", ".json")
            json_path = os.path.join(DATASET_DIR, json_name)
            
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    meta = json.load(f)
                
                true_dist = meta.get("distance_m", 0.0)
                true_roll = meta.get("roll_deg", 0.0)
                true_pitch = meta.get("pitch_deg", 0.0)
                true_yaw = meta.get("yaw_deg", 0.0)
                
                frames.append(count)
                true_dists.append(true_dist)
                est_dists.append(est_dist)
                true_rolls.append(true_roll)
                est_rolls.append(est_roll)
                true_pitchs.append(true_pitch)
                est_pitchs.append(est_pitch)
                true_yaws.append(true_yaw)
                est_yaws.append(est_yaw)
                execution_times.append(exec_ms)

process.wait()
print("\n")

if not execution_times:
    print("❌ Aucune donnée valide reçue du binaire Rust.")
    sys.exit(1)

def angular_mae(y_true, y_pred):
    diff = np.abs(np.array(y_true) - np.array(y_pred))
    return np.mean(np.where(diff > 180, 360 - diff, diff))

mae_dist = np.mean(np.abs(np.array(est_dists) - np.array(true_dists)))
mae_roll = angular_mae(true_rolls, est_rolls)
mae_pitch = angular_mae(true_pitchs, est_pitchs)
mae_yaw = angular_mae(true_yaws, est_yaws)

print("📊 Génération des graphiques...")
fig, axs = plt.subplots(4, 1, figsize=(12, 16))

axs[0].plot(frames, true_dists, label="Vérité Terrain (Simulateur)", color="black", linestyle="--")
axs[0].plot(frames, est_dists, label="Estimation Réalignée (Rust)", color="blue", alpha=0.7)
axs[0].set_title(f"Distance relative Drone-Marqueur (MAE: {mae_dist:.3f} m)", fontsize=13)
axs[0].set_ylabel("Distance (m)")
axs[0].legend()
axs[0].grid(True, linestyle=":", alpha=0.6)

axs[1].plot(frames, true_rolls, label="Vérité Terrain", color="black", linestyle="--")
axs[1].plot(frames, est_rolls, label="Estimation Réalignée", color="red", alpha=0.7)
axs[1].set_title(f"Suivi du Roulis / Roll (MAE: {mae_roll:.2f}°)", fontsize=13)
axs[1].set_ylabel("Angle (°)")
axs[1].legend()
axs[1].grid(True, linestyle=":", alpha=0.6)

axs[2].plot(frames, true_pitchs, label="Vérité Terrain", color="black", linestyle="--")
axs[2].plot(frames, est_pitchs, label="Estimation Réalignée", color="green", alpha=0.7)
axs[2].set_title(f"Suivi du Tangage / Pitch (MAE: {mae_pitch:.2f}°)", fontsize=13)
axs[2].set_ylabel("Angle (°)")
axs[2].legend()
axs[2].grid(True, linestyle=":", alpha=0.6)

axs[3].plot(frames, true_yaws, label="Vérité Terrain", color="black", linestyle="--")
axs[3].plot(frames, est_yaws, label="Estimation Réalignée", color="purple", alpha=0.7)
axs[3].set_title(f"Suivi du Lacet / Yaw (MAE: {mae_yaw:.2f}°)", fontsize=13)
axs[3].set_xlabel("Nombre de Frames du Dataset")
axs[3].set_ylabel("Angle (°)")
axs[3].legend()
axs[3].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
output_file = "benchmark_performance_dashboard.png"
plt.savefig(output_file, dpi=200)
print(f"✅ Tableau de bord haute résolution sauvegardé sous : {output_file}\n")

print("==================================================")
print("🎯 BILAN DES ERREURS MOYENNES RÉELLES (MAE)")
print("==================================================")
print(f"⚡ Vitesse          : {np.mean(execution_times):.3f} ms/frame ({1000.0/np.mean(execution_times):.1f} FPS)")
print(f"📏 Distance Z       : {mae_dist:.4f} m")
print(f"📐 Roll (Roulis)    : {mae_roll:.3f}°")
print(f"📐 Pitch (Tangage)  : {mae_pitch:.3f}°")
print(f"📐 Yaw (Lacet)      : {mae_yaw:.3f}°")
print("==================================================")
