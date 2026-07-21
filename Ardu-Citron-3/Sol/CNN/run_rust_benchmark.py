#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import glob
import numpy as np
import cv2
import core_localizer_rust

def load_dataset_sorted(dataset_path):
    """Charge et trie les images et les fichiers JSON du dataset."""
    image_files = sorted(glob.glob(os.path.join(dataset_path, "*.png"))) + \
                  sorted(glob.glob(os.path.join(dataset_path, "*.jpg")))
    
    dataset = []
    for img_path in image_files:
        base_name = os.path.splitext(img_path)[0]
        json_path = base_name + ".json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                gt_data = json.load(f)
            dataset.append((img_path, gt_data))
    return dataset

def rvec_to_euler(rvec):
    """Convertit un vecteur de rotation Rodrigues en angles d'Euler (Roll, Pitch, Yaw)."""
    rmat, _ = cv2.Rodrigues(rvec)
    yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
    pitch = np.arctan2(-rmat[2, 0], np.sqrt(rmat[2, 1]**2 + rmat[2, 2]**2))
    roll = np.arctan2(rmat[2, 1], rmat[2, 2])
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def run_benchmark():
    DATASET_PATH = "/home/sts33/2eme/ODB/Ardu-Citron-3/Sol/Markers_5/Dataset"
    
    dataset = load_dataset_sorted(DATASET_PATH)
    if not dataset:
        print(f"Erreur : Aucun couple Image/JSON trouvé dans : {DATASET_PATH}")
        return

    print(f"📊 Dataset chargé : {len(dataset)} images trouvées.")
    print("🏃 Lancement du calcul global sur TOUTES les trajectoires...")
    
    # Configuration caméra & ArUco
    fx, fy, cx, cy = 492.72, 492.72, 320.0, 240.0
    cam_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.07, 0, 0, 0, 0], dtype=np.float32)
    
    marker_size = 0.20
    h = marker_size / 2.0
    obj_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Instanciation du moteur mathématique Rust
    engine = core_localizer_rust.DroneFusionEngine()

    # Listes globales pour le calcul des erreurs RMS finales
    all_gt_dist, all_rust_dist = [], []
    all_gt_roll, all_rust_roll = [], []
    all_gt_pitch, all_rust_pitch = [], []
    all_gt_yaw, all_rust_yaw = [], []
    
    processing_times = []
    last_timestamp = None
    last_center = None
    last_trajectory = None
    roi_half_size = 100

    for idx, (img_path, gt_data) in enumerate(dataset):
        current_trajectory = gt_data.get("trajectory_id")
        
        # SÉCURITÉ : Si on change de trajectoire au milieu du dataset, 
        # on réinitialise les filtres de Kalman et la ROI pour ne pas fausser les calculs !
        if current_trajectory != last_trajectory:
            engine.reset()
            last_center = None
            last_timestamp = None
            last_trajectory = current_trajectory

        frame = cv2.imread(img_path)
        if frame is None: continue
        h_img, w_img, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        current_timestamp = gt_data.get("timestamp_s", idx * 0.033)
        dt = current_timestamp - last_timestamp if last_timestamp is not None else 0.033
        last_timestamp = current_timestamp
        
        imu_pitch = gt_data.get("imu_mpu6050", {}).get("pitch_deg", 0.0)

        t_start = time.perf_counter()

        # Tracking ROI
        offset_x, offset_y = 0, 0
        search_img = gray
        if last_center is not None:
            cx, cy = last_center
            x1, y1 = max(0, cx - roi_half_size), max(0, cy - roi_half_size)
            x2, y2 = min(w_img, cx + roi_half_size), min(h_img, cy + roi_half_size)
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                search_img = gray[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1

        corners, ids, _ = detector.detectMarkers(search_img)

        if ids is not None and len(ids) > 0:
            marker_corners = corners[0][0]
            if offset_x > 0 or offset_y > 0:
                marker_corners[:, 0] += offset_x
                marker_corners[:, 1] += offset_y

            cx_new, cy_new = int(np.mean(marker_corners[:, 0])), int(np.mean(marker_corners[:, 1]))
            last_center = (cx_new, cy_new)

            # PnP
            _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_points, marker_corners, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            
            best_rvec, best_tvec = rvecs[0], tvecs[0]
            if len(tvecs) > 1:
                _, p1, _ = rvec_to_euler(rvecs[0])
                _, p2, _ = rvec_to_euler(rvecs[1])
                if abs((p2 - imu_pitch + 180) % 360 - 180) < abs((p1 - imu_pitch + 180) % 360 - 180):
                    best_rvec, best_tvec = rvecs[1], tvecs[1]

            cv2.solvePnP(obj_points, marker_corners, cam_matrix, dist_coeffs, rvec=best_rvec, tvec=best_tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

            raw_d = float(np.linalg.norm(best_tvec))
            raw_r, raw_p, raw_y = rvec_to_euler(best_rvec)

            # Moteur Rust Kalman
            pose = engine.fuse_and_filter(raw_d, raw_r, raw_p, raw_y, dt, imu_pitch)
            
            processing_times.append((time.perf_counter() - t_start) * 1000.0)

            # Stockage des données pour le calcul final
            all_gt_dist.append(gt_data.get("distance_m", 0.0))
            all_gt_roll.append(gt_data.get("roll_deg", 0.0))
            all_gt_pitch.append(gt_data.get("pitch_deg", 0.0))
            all_gt_yaw.append(gt_data.get("yaw_deg", 0.0))
            
            all_rust_dist.append(pose["distance"])
            all_rust_roll.append(pose["roll"])
            all_rust_pitch.append(pose["pitch"])
            all_rust_yaw.append(pose["yaw"])
        else:
            last_center = None

        # Un petit indicateur de progression car 7500 images c'est long
        if idx % 1500 == 0 and idx > 0:
            print(f" ⏳ Progression : {idx}/{len(dataset)} images traitées...")

    # --- CALCULS DES MÉTRIQUES FINALES ---
    mean_lat = np.mean(processing_times)
    rms_dist = np.sqrt(np.mean((np.array(all_gt_dist) - np.array(all_rust_dist))**2))
    rms_roll = np.sqrt(np.mean((np.array(all_gt_roll) - np.array(all_rust_roll))**2))
    rms_pitch = np.sqrt(np.mean((np.array(all_gt_pitch) - np.array(all_rust_pitch))**2))
    rms_yaw = np.sqrt(np.mean((np.array(all_gt_yaw) - np.array(all_rust_yaw))**2))

    print("\n==========================================================")
    print("🏆 RAPPORT DE PERFORMANCE GLOBAL (7500 IMAGES MULTI-TRAJ)")
    print("==========================================================")
    print(f"⚡ Temps de calcul moyen/image : {mean_lat:.3f} ms ({1000.0 / mean_lat:.1f} Hz)")
    print(f"📏 Erreur RMS DISTANCE         : {rms_dist:.4f} m  (soit {rms_dist*100:.1f} cm)")
    print(f"🔄 Erreur RMS ROLL (Roulis)   : {rms_roll:.3f} °")
    print(f"📐 Erreur RMS PITCH (Tangage) : {rms_pitch:.3f} °")
    print(f"🧭 Erreur RMS YAW (Lacet)     : {rms_yaw:.3f} °")
    print("==========================================================\n")

if __name__ == "__main__":
    run_benchmark()
