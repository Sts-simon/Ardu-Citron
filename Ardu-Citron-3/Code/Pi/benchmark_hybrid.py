import os
import subprocess
import json
import numpy as np
import sys

DATASET_DIR = "/mnt/1to/ODB/Ardu-Citron-3/Sol/Markers_5/Dataset"
RUST_BIN = "./drone_localizer/target/release/drone_localizer"
TOTAL_IMAGES = 500  

print(f"🚀 Lancement du benchmark corrigé selon audit (Sub-pixel + Rodrigues + Check Biais)...")

process = subprocess.Popen(
    [RUST_BIN, DATASET_DIR, str(TOTAL_IMAGES)],
    stdout=subprocess.PIPE,
    text=True
)

errors_norm_3d = []
errors_z_pure = []
errors_pitch = []
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
            filename = parts[1]
            rust_norm_3d = float(parts[2])
            rust_z_pure = float(parts[3])
            rust_pitch = float(parts[4])
            rust_time_ms = float(parts[5])
            
            execution_times.append(rust_time_ms)
            
            json_path = os.path.join(DATASET_DIR, filename.replace(".png", ".json"))
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    truth = json.load(f)
                
                true_dist = truth["distance_m"]
                true_pitch = truth["pitch_deg"] # 🚀 On utilise la vraie valeur signée !
                
                # POINT 1 : On évalue les deux méthodes pour détecter le biais systématique
                errors_norm_3d.append(abs(rust_norm_3d - true_dist))
                errors_z_pure.append(abs(rust_z_pure - true_dist))
                errors_pitch.append(abs(rust_pitch - true_pitch))

process.wait()
print("\n")

print("==================================================")
print("🎯 RÉSULTATS DU SYSTÈME APRÈS AUDIT")
print("==================================================")
if execution_times:
    print(f"⚡ Temps calcul (Sub-pixel)      : {np.mean(execution_times):.3f} ms / image")
    print(f"🏎️  Cadence théorique PC          : {1000.0 / np.mean(execution_times):.1f} FPS")
    print("--------------------------------------------------")
    print(f"📏 Erreur si GT = Norme 3D (MAE) : {np.mean(errors_norm_3d):.4f} mètres")
    print(f"📏 Erreur si GT = Axe Z pur (MAE): {np.mean(errors_z_pure):.4f} mètres")
    print("--------------------------------------------------")
    print(f"📐 Précision Pitch Signé (MAE)   : {np.mean(errors_pitch):.3f} degrés")
else:
    print("❌ Aucune donnée collectée.")
print("==================================================\n")
