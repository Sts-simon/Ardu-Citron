import os
import glob
import json
import cv2
import torch
import numpy as np
import time
from train_cnn import TinyDroneLocalizer

DATA_DIR = "./cnn_roi_dataset"
NUM_TEST_IMAGES = 10000  # Nombre d'images à tester pour les statistiques

def calculate_statistics():
    # 1. Initialiser le modèle
    model = TinyDroneLocalizer()
    model_path = "tiny_drone_epoch_14.pth"
    if not os.path.exists(model_path):
        model_path = "checkpoints/tiny_drone_cnn.pth"
        
    if not os.path.exists(model_path):
        print("❌ Impossible de trouver 'tiny_drone_cnn.pth'.")
        return

    # Chargement des poids
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 2. Récupérer les images pour le test
    image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))
    if len(image_paths) == 0:
        print(f"❌ Aucune image trouvée dans {DATA_DIR}.")
        return
    
    image_paths = image_paths[:min(NUM_TEST_IMAGES, len(image_paths))]
    print(f"🔬 Analyse statistique et benchmark en cours sur {len(image_paths)} images...")

    # Listes pour stocker les erreurs et les temps
    errors_pos_cm = []
    errors_roll = []
    errors_pitch = []
    errors_yaw = []
    inference_times_ms = []

    # 3. Boucle d'inférence avec chronomètre
    with torch.no_grad():
        for img_path in image_paths:
            json_path = img_path.replace(".png", ".json")
            if not os.path.exists(json_path):
                continue
                
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            # ⏱️ Début du chrono (On ne mesure QUE le calcul du CNN)
            start_time = time.perf_counter()
            output = model(img_tensor).squeeze(0).numpy()
            end_time = time.perf_counter()
            
            # Calcul du temps en millisecondes
            elapsed_ms = (end_time - start_time) * 1000.0
            inference_times_ms.append(elapsed_ms)
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            gt_norm = data["target_pose_xyz_rpy"]
            
            # Dénormalisation
            pred_x, pred_y, pred_z = output[0], output[1], output[2]
            pred_r, pred_p, pred_y_deg = output[3]*180.0, output[4]*180.0, output[5]*180.0
            
            gt_x, gt_y, gt_z = gt_norm[0], gt_norm[1], gt_norm[2]
            gt_r, gt_p, gt_y_deg = gt_norm[3], gt_norm[4], gt_norm[5]
            
            # Calcul des erreurs absolues
            err_dist_cm = np.sqrt((pred_x-gt_x)**2 + (pred_y-gt_y)**2 + (pred_z-gt_z)**2) * 100.0
            
            errors_pos_cm.append(err_dist_cm)
            errors_roll.append(abs(pred_r - gt_r))
            errors_pitch.append(abs(pred_p - gt_p))
            errors_yaw.append(abs(pred_y_deg - gt_y_deg))

    # 4. Calcul des métriques statistiques globales
    def print_metric_stats(name, unit, data_list):
        arr = np.array(data_list)
        print(f"📊 {name:<18} | Moyenne: {np.mean(arr):6.2f}{unit} | Médiane: {np.median(arr):6.2f}{unit} | Max: {np.max(arr):6.2f}{unit}")

    avg_time = np.mean(inference_times_ms)
    fps = 1000.0 / avg_time

    print("\n========================================================================")
    print(f"📈 RAPPORT DE PRÉCISION & BENCHMARK (Échantillon : {len(errors_pos_cm)} images)")
    print("========================================================================")
    print_metric_stats("Position Globale", "cm", errors_pos_cm)
    print_metric_stats("Erreur Roll", "° ", errors_roll)
    print_metric_stats("Erreur Pitch", "° ", errors_pitch)
    print_metric_stats("Erreur Yaw (Cap)", "° ", errors_yaw)
    print("------------------------------------------------------------------------")
    print(f"⚡ Temps d'inférence moyen : {avg_time:.2f} ms par image")
    print(f"🚀 Vitesse estimée         : {fps:.1f} FPS (sur ce CPU)")
    print("========================================================================")

if __name__ == "__main__":
    calculate_statistics()
