import os
import glob
import json
import cv2
import numpy as np
import time
import onnxruntime as ort

DATA_DIR = "./cnn_roi_dataset"
NUM_TEST_IMAGES = 1000  # Échantillon de test
ONNX_PATH = "tiny_drone_localizer.onnx"

def calculate_onnx_statistics():
    if not os.path.exists(ONNX_PATH):
        print(f"❌ Impossible de trouver le fichier '{ONNX_PATH}'. Lance d'abord train_cnn.py pour le générer !")
        return

    # 1. Initialiser la session de calcul ONNX Runtime sur CPU
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # 2. Récupérer les images pour le test
    image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))
    if len(image_paths) == 0:
        print(f"❌ Aucune image trouvée dans {DATA_DIR}.")
        return

    image_paths = image_paths[:min(NUM_TEST_IMAGES, len(image_paths))]
    print(f"🔬 ONNX Benchmark & Analyse (Nouvelle Normalisation) en cours sur {len(image_paths)} images...")

    # Listes pour stocker les erreurs et les temps
    errors_pos_cm = []
    errors_roll = []
    errors_pitch = []
    errors_yaw = []
    inference_times_ms = []

    # 3. Boucle de test
    for img_path in image_paths:
        json_path = img_path.replace(".png", ".json")
        if not os.path.exists(json_path):
            continue

        # Lecture du Ground Truth (Vraies valeurs physiques)
        with open(json_path, 'r') as f:
            data = json.load(f)
        gt_pose = data["target_pose_xyz_rpy"]
        gt_x, gt_y, gt_z = gt_pose[0], gt_pose[1], gt_pose[2]
        gt_r, gt_p, gt_y_deg = gt_pose[3], gt_pose[4], gt_pose[5]

        # Prétraitement de l'image (Identique à l'entraînement)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # [H, W, C] -> [C, H, W] et normalisation [0, 1]
        img_input = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        # Ajouter la dimension du batch [1, C, H, W]
        img_input = np.expand_dims(img_input, axis=0)

        # Inférence et mesure du temps
        start_time = time.time()
        outputs = session.run(None, {input_name: img_input})
        elapsed_ms = (time.time() - start_time) * 1000.0
        inference_times_ms.append(elapsed_ms)

        # Récupération de la sortie brute du modèle ONNX (6 valeurs normalisées)
        pred_norm = outputs[0][0]

        # ✨ APPLICATION DE LA DÉ-NORMALISATION INVERSE
        # (Pour faire correspondre les sorties du réseau aux vraies unités physiques)
        pred_x = pred_norm[0] * 3.0
        pred_y = pred_norm[1] * 3.0
        pred_z = (pred_norm[2] * (6.0 - 2.0)) + 2.0  # Restaure l'altitude Z entre 2m et 6m
        
        pred_r = pred_norm[3] * 35.0   # Restaure le Roll (max 35°)
        pred_p = pred_norm[4] * 20.0   # Restaure le Pitch (max 20°)
        pred_y_deg = pred_norm[5] * 45.0 # Restaure le Yaw (max 45°)

        # Calcul des erreurs absolues
        err_dist_cm = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2 + (pred_z - gt_z)**2) * 100.0
        
        errors_pos_cm.append(err_dist_cm)
        errors_roll.append(abs(pred_r - gt_r))
        errors_pitch.append(abs(pred_p - gt_p))
        errors_yaw.append(abs(pred_y_deg - gt_y_deg))

    # 4. Affichage du rapport comparatif
    def print_metric_stats(name, unit, data_list):
        arr = np.array(data_list)
        print(f"📊 {name:<18} | Moyenne: {np.mean(arr):6.2f}{unit} | Médiane: {np.median(arr):6.2f}{unit} | Max: {np.max(arr):6.2f}{unit}")

    avg_time = np.mean(inference_times_ms)
    fps = 1000.0 / avg_time

    print("\n========================================================================")
    print(f"⚡ RAPPORT DE PRÉCISION & VITESSE *ONNX RUNTIME* ({len(errors_pos_cm)} images)")
    print("========================================================================")
    print_metric_stats("Position Globale", "cm", errors_pos_cm)
    print_metric_stats("Erreur Roll", "° ", errors_roll)
    print_metric_stats("Erreur Pitch", "° ", errors_pitch)
    print_metric_stats("Erreur Yaw", "° ", errors_yaw)
    print("------------------------------------------------------------------------")
    print(f"⏱️  Temps d'inférence CPU moyen : {avg_time:.2f} ms | Fréquence estimée : {fps:.1f} FPS")
    print("========================================================================\n")

if __name__ == "__main__":
    calculate_onnx_statistics()
