import os
import json
import glob
import cv2
import numpy as np

# Configuration des chemins
DATASET_DIR = "./Dataset-2" 
OUTPUT_DIR = "./cnn_roi_dataset-2"
ROI_SIZE = (128, 128) # Taille fixe d'entrée pour le CNN
AUGMENTATION_FACTOR = 5 # Nombre de variantes générées par image originale

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_augmented_roi_dataset():
    image_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.png")))
    print(f"🔍 Trouvé {len(image_paths)} images originales.")

    if len(image_paths) == 0:
        print("❌ Aucune image trouvée. Vérifie le dossier './Dataset'.")
        return

    count = 0
    for img_path in image_paths:
        json_path = img_path.replace(".png", ".json")
        if not os.path.exists(json_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img, _ = img.shape

        with open(json_path, 'r') as f:
            data = json.load(f)

        try:
            x = data.get("pos_x", data.get("x", 0.0))
            y = data.get("pos_y", data.get("y", 0.0))
            z = data.get("pos_z", data.get("z", data.get("distance", 0.0)))
            
            gt_labels = [
                x, y, z,
                data["roll_deg"], data["pitch_deg"], data["yaw_deg"]
            ]
            corners = data.get("aruco_corners", None)
        except KeyError as e:
            continue

        # Détection ArUco si pas dans le JSON
        if corners is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            try:
                dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                parameters = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(dictionary, parameters)
                detected_corners, ids, _ = detector.detectMarkers(gray)
            except AttributeError:
                dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
                parameters = cv2.aruco.DetectorParameters_create()
                detected_corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

            if ids is None or len(detected_corners) == 0:
                continue
            corners = detected_corners[0][0]

        # Calcul de la boîte englobante
        x_coords = [c[0] for c in corners]
        y_coords = [c[1] for c in corners]
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))
        
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        box_w = xmax - xmin
        box_h = ymax - ymin
        
        # 🔥 MODIFICATION : Marge augmentée de 0.4 à 1.2 pour inclure l'environnement du drone
        margin = int(max(box_w, box_h) * 1.2)

        for i in range(AUGMENTATION_FACTOR):
            if i == 0:
                dx, dy = 0, 0
                scale_modifier = 1.0
            else:
                # Bruit de tracking modéré pour éviter de sortir le marqueur de la vue large
                dx = np.random.randint(-20, 20) 
                dy = np.random.randint(-20, 20) 
                scale_modifier = np.random.uniform(1.0, 1.4) # Donne la priorité aux plans larges (zoom arrière)

            current_margin = int(margin * scale_modifier)
            
            # Application du Crop avec sécurité sur les bordures de la frame d'origine
            crop_xmin = max(0, center_x - current_margin + dx)
            crop_xmax = min(w_img, center_x + current_margin + dx)
            crop_ymin = max(0, center_y - current_margin + dy)
            crop_ymax = min(h_img, center_y + current_margin + dy)

            roi = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            if roi.size == 0:
                continue
            roi_resized = cv2.resize(roi, ROI_SIZE)

            base_name = os.path.basename(img_path).replace(".png", "")
            new_img_name = f"{base_name}_aug_{i}.png"
            new_json_name = f"{base_name}_aug_{i}.json"

            cv2.imwrite(os.path.join(OUTPUT_DIR, new_img_name), roi_resized)

            output_data = {
                "img_channels_height_width": [3, ROI_SIZE[0], ROI_SIZE[1]],
                "target_pose_xyz_rpy": gt_labels
            }
            
            with open(os.path.join(OUTPUT_DIR, new_json_name), 'w') as out_f:
                json.dump(output_data, out_f, indent=4)
            
            count += 1

    print(f"✅ Dataset ROI (Zone Large) généré dans '{OUTPUT_DIR}' : {count} images.")

if __name__ == "__main__":
    create_augmented_roi_dataset()
