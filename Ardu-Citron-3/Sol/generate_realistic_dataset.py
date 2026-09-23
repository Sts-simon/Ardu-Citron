#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_realistic_dataset.py — scenes ArUco realistes, images independantes.

Derive de generate_dataset.py (Ardu-Citron). Toute la physique de rendu est
reprise a l'identique (texture de sol par homographie, ombre portee, flou de
mouvement, rolling shutter, distorsion optique, profondeur de champ, reflets
speculaires, vibrations, neon + balance des blancs, vignetage, bruit capteur,
JPEG, imperfections physiques du marqueur). Deux dependances du script
d'origine sont retirees :

  * cairosvg + PIL + fichiers 4x4_1000-*.svg  -> generate_marker_rgba() genere
    le marqueur directement avec cv2.aruco (DICT_4X4_1000, coherent avec le
    nommage 4x4_1000-N.svg de l'original) ;
  * aruco_detector.ArucoDetector (module non fourni, servait uniquement a
    logger un taux de detection pendant la generation) -> retire ; la
    validation se fait a part, avec aruco_dedup_v2.py, sur le dataset produit.

La partie "trajectoire de vol + IMU" (process_verification_marker) n'est pas
reprise : elle simule 500 images/s d'un meme vol, ce que l'usage present
(generer des scenes de TEST variees pour le detecteur) ne demande pas.
render_independent_scene(), plus bas, remplace generate_cnn_examples_for_marker :
meme tirage aleatoire independant (pose, texture, eclairage, imperfections),
mais sort l'IMAGE PLEINE (640x480), pas un crop 128x128 recadre sur un seul
marqueur — et place PLUSIEURS marqueurs par scene, avec la meme contrainte de
visibilite minimale que l'original (cnn_min_visible_fraction / _px).

Chaque image est accompagnee d'un JSON verite-terrain : id, coins pixel,
fraction visible, et les parametres de degradation tires pour cette image
(utile pour analyser une baisse de rappel par cause : flou, distance,
vignetage, etc.)

Usage :
    python3 generate_realistic_dataset.py --n 60 --markers-per-scene 1-3 \
        --out dataset_realiste --seed 0
    python3 generate_realistic_dataset.py --n 20 --difficulty hard --out hard/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Tuple

import cv2
import numpy as np

CONFIG = {
    "trajectory_duration_s": 1.0,   # Durée d'une trajectoire continue (en secondes)
    "frames_per_trajectory": 500,   # Nb d'images par trajectoire -> 1 trajectoire = 1s = 500 images
    "output_resolution": (640, 480), # Résolution de la caméra (Largeur, Hauteur)
    
    # Paramètres de vol du drone (aile fixe)
    "altitude_min": 2.0,           # en mètres
    "altitude_max": 6.0,           # en mètres
    "drone_speed": 8.0,            # en m/s (vitesse air, quasi constante en aile fixe)

    # Dynamique de vol -> génère des trajectoires cohérentes (pas de saut aléatoire frame à frame)
    "roll_max_deg": 35.0,           # Inclinaison max en virage stabilisé
    "pitch_max_deg": 20.0,          # Assiette max (montée/descente)
    "yaw_rate_range_deg_s": (8.0, 32.0),    # Vitesse de lacet en virage (deg/s) -> virages plus francs
    "climb_rate_range_ms": (0.4, 2.0),      # Taux de montée/descente (m/s)
    "roll_lag_tau_s": 0.22,         # Constante de temps du roulis (inertie/actionneur) -> virage progressif
    "turbulence_sigma_deg": 1.8,    # Amplitude du bruit de turbulence "lent" sur les angles (deg)
    "turbulence_tau_s": 0.15,       # Constante de temps du bruit de turbulence lent (corrélation temporelle)
    "turbulence_fast_sigma_deg": 0.9,  # Amplitude du bruit "rapide" (jitter/rafales courtes) superposé au lent
    "turbulence_fast_tau_s": 0.045,    # Constante de temps du bruit rapide -> plus de mouvement haute fréquence
    "altitude_turbulence_sigma_m": 0.07, # Amplitude des rafales verticales (m)
    "s_turn_half_cycles_choices": [1, 2, 2, 3],  # Nb d'inversions de virage pour les manœuvres en S
    "wave_cycles_range": (1.0, 2.5),   # Nb d'oscillations pour la manœuvre "vague" (façon phugoïde)

    # Simulation IMU embarquée (MPU6050 : gyroscope 3 axes + accéléromètre 3 axes)
    "imu_gyro_noise_density_dps": 0.03,      # Bruit blanc gyro (°/s) -> bruit de mesure haute fréquence
    "imu_gyro_bias_init_range_dps": (-3.0, 3.0),   # Biais gyro initial aléatoire (non calibré à l'allumage)
    "imu_gyro_bias_walk_sigma_dps": 0.05,    # Amplitude de la dérive lente du biais gyro (random walk)
    "imu_gyro_bias_walk_tau_s": 4.0,         # Constante de temps de la dérive du biais gyro
    "imu_accel_noise_sigma_dps": 0.6,        # Bruit sur l'angle déduit de l'accéléromètre (vibrations, ADC)
    "imu_accel_roll_attenuation": 0.35,      # Atténuation du roulis "vu" par l'accéléro en virage coordonné
    "imu_initial_attitude_error_deg": (2.0, 6.0),  # Erreur d'attitude initiale aléatoire à la 1ère frame
    "imu_complementary_alpha": 0.98,         # Coefficient du filtre complémentaire (0.98 gyro / 0.02 accéléro)
    
    # Caractéristiques physiques et optiques
    "marker_real_size": 0.20,      # Taille réelle du marqueur (0.20m x 0.20m) pour cohérence avec le benchmark
    "camera_h_fov": 66.0,          # FOV Horizontal Raspberry Pi Cam v3 (IMX708) en degrés
    "camera_v_fov": 52.0,          # FOV Vertical en degrés
    "exposure_time": 1.0 / 500.0,   # Temps de pose de la caméra (en secondes)
    "rolling_shutter_readout": 0.02, # Temps de balayage du capteur (en secondes)
    "k1_distortion": -0.07,        # (conservé pour compatibilité) Coefficient radial k1 de la lentille (IMX708)

    # --- Géométrie de vol : translation réelle du drone (dérive du marqueur dans l'image) ---
    "position_drift_scale": 1.0,     # Facteur d'échelle sur la vitesse horizontale intégrée (X,Y du drone)
    "wind_gust_sigma_m": 0.35,       # Amplitude de la dérive latérale due au vent (Random Walk, en mètres)
    "wind_gust_tau_s": 0.40,         # Constante de temps de la dérive du vent (corrélation lente)

    # --- Entrée / sortie de cadre : le marqueur apparaît d'un côté et traverse l'image ---
    "trajectories_per_marker": 4,         # Nb de trajectoires (vols) générées par marqueur (dataset VÉRIFICATION)
    "entry_touch_factor_range": (0.65, 0.9),  # Position d'entrée (fraction du rayon d'empreinte au sol)
    "entry_angle_jitter_deg": 45.0,      # Dispersion angulaire de l'entrée autour du cap moyen

    # --- Rendu du sol : textures hétérogènes + homographie complète ---
    "ground_texture_types": ["gym_floor", "wood", "tile", "concrete", "grass", "asphalt", "dirt"],
    "ground_texture_weights": [0.55, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04],  # gymnase privilégié
    "ground_texture_size_m": 24.0,   # Taille physique du patch de texture généré (mètres, carré)
    "ground_texels_per_meter": 50,   # Résolution de la texture (pixels de texture par mètre réel)

    # --- Lignes de terrain (gymnase) : couleurs, épaisseur, éléments dessinés ---
    "gym_line_colors_bgr": {
        "white": (235, 235, 235),
        "yellow": (40, 210, 235),
        "red": (55, 55, 195),
        "blue": (195, 110, 50),
    },
    "gym_court_line_width_px_range": (4, 9),
    "gym_logo_prob": 0.6,           # Probabilité d'ajouter un logo/cercle central stylisé
    "gym_number_prob": 0.5,         # Probabilité d'ajouter des numéros peints au sol

    # --- Reflets du parquet (spéculaire) ---
    "specular_highlight_count_range": (0, 3),
    "specular_highlight_intensity_range": (60, 160),
    "specular_highlight_length_frac_range": (0.15, 0.45),  # fraction de la diagonale image

    # --- Éclairage : soleil (extérieur), néons (gymnase), balance des blancs, AE, vignetage ---
    "sun_elevation_range_deg": (25.0, 75.0),   # Hauteur du soleil dans le ciel (degrés)
    "shadow_length_coeff": 0.12,     # Coefficient reliant altitude drone -> longueur d'ombre projetée
    "shadow_blur_base_px": 5,        # Flou de base de l'ombre (pixels)
    "shadow_blur_altitude_coeff": 3.0,  # L'ombre devient plus floue (pénombre) quand l'altitude augmente
    "neon_flicker_freq_hz": 100.0,     # Fréquence de scintillement des tubes néon/LED (secteur redressé)
    "neon_flicker_amplitude_range": (0.03, 0.10),
    "neon_green_tint_range": (0.0, 0.08),   # Dominante verte typique des tubes fluorescents
    "wb_temp_range": (-1.0, 1.0),     # Balance des blancs : -1 froid, +1 chaud
    "wb_green_range": (-0.2, 0.3),    # Composante verte additionnelle de la balance des blancs
    "wb_drift_sigma": 0.35,           # Amplitude de dérive lente de la WB pendant la trajectoire
    "wb_drift_tau_s": 0.4,            # Constante de temps de la dérive de WB
    "ae_gain_range": (0.7, 1.3),      # Gain d'auto-exposition appliqué frame à frame
    "vignette_strength_range": (0.15, 0.35),  # Intensité du vignetage optique

    # --- Bruit capteur réaliste (IMX708) & profondeur de champ ---
    "noise_luma_sigma_range": (2, 8),     # Bruit gaussien sur la luminance (Y)
    "noise_chroma_sigma_range": (4, 14),  # Bruit plus fort sur la chrominance (Cr/Cb), typique petits capteurs
    "chroma_lowlight_boost": 1.5,         # Amplification du bruit chroma dans les zones sombres
    "shot_noise_coeff": 0.35,             # Bruit de photon (Poisson) : sigma ∝ sqrt(intensité)
    "hot_pixel_count_range": (0, 4),      # Nb de pixels chauds (défauts capteur, fixes par trajectoire)
    "hot_pixel_value": 255,
    "focus_error_range_m": (-1.0, 1.0),   # Erreur de mise au point autofocus par rapport à l'altitude initiale
    "dof_blur_base_px": 3,                # Flou de base (mise au point parfaite)
    "dof_blur_coeff_px_per_m": 2.0,       # Flou additionnel par mètre d'écart à la distance de mise au point
    "dof_max_ksize": 11,                  # Taille max du noyau de flou (px)
    "autofocus_hunt_event_prob": 0.15,    # Probabilité d'un "saut" de mise au point pendant la trajectoire
    "autofocus_hunt_len_frames_range": (15, 60),  # Durée (en frames) d'un saut de mise au point
    "autofocus_hunt_extra_ksize": 6,      # Flou additionnel pendant un saut de mise au point

    # --- Vibrations mécaniques (moteur/hélice/servos) ---
    "vibration_amplitude_px_range": (0.3, 1.2),   # Amplitude du tremblement image par image
    "vibration_blur_coeff": 2.0,                  # Flou de mouvement additionnel induit par la vibration

    # --- Distorsion optique complète (Brown-Conrady : radiale k1,k2,k3 + tangentielle p1,p2) ---
    "k1_distortion_range": (-0.14, -0.04),
    "k2_distortion_range": (-0.02, 0.02),
    "k3_distortion_range": (-0.01, 0.01),
    "p1_distortion_range": (-0.004, 0.004),
    "p2_distortion_range": (-0.004, 0.004),

    # --- Compression JPEG (artefacts de codec appliqués avant sauvegarde) ---
    "jpeg_quality_choices": [70, 75, 80, 85, 90, 95],

    # --- Marqueur imparfait (papier réel, impression, gondolement) ---
    "marker_black_level_range": (10, 45),    # Le "noir" du marqueur n'est jamais parfaitement noir
    "marker_white_level_range": (200, 245),  # Le "blanc" n'est jamais parfaitement blanc
    "marker_paper_noise_sigma": 4.0,         # Grain du papier
    "marker_warp_prob": 0.5,                 # Probabilité d'un léger gondolement (papier non plan)
    "marker_warp_amplitude_px_range": (1.0, 4.0),
    "marker_corner_lift_prob": 0.35,         # Probabilité d'un coin décollé/froissé (assombrissement local)

    # Intensité des effets
    "autofocus_blur_prob": 0.3,    # (conservé pour compatibilité, non utilisé par le nouveau pipeline DOF)

    # Chemins des fichiers
    "verification_output_dir": "Dataset_Verification",
    "markers_dir": "Markers_5",

    # --- Dataset CNN (train/val) : exemples indépendants, sans trajectoire (diversité maximale) ---
    "cnn_output_dir": "Dataset_CNN",
    "cnn_examples_per_marker": 500,   # Nb d'exemples indépendants générés par marqueur
    "cnn_val_fraction": 0.15,         # Fraction réservée à la validation (85% train / 15% val)
    "roi_size": (128, 128),           # Taille fixe d'entrée pour le CNN
    "roi_margin_factor": 1.2,         # Marge de base autour du marqueur (contexte visuel)
    "roi_scale_range": (1.0, 1.6),    # Variation de zoom arrière du crop (plans plus ou moins larges)
    "roi_position_jitter_frac": 0.6,  # Décentrage aléatoire du marqueur dans le crop (fraction de la marge)

    # --- Garantie de visibilité du marqueur (dataset CNN) ---
    "cnn_min_visible_fraction": 0.2,   # Fraction mini de la bbox du marqueur qui doit être dans le cadre
    "cnn_min_visible_px": 12,          # Taille mini (px) de la portion visible, largeur ET hauteur
    "cnn_placement_max_attempts": 25,  # Nb de re-tirages de position avant d'abandonner cet exemple
}

def _multiscale_noise(h, w, scales_sigmas):
    """Bruit à plusieurs échelles spatiales (basse fréquence = mottling, haute fréquence = grain)."""
    total = np.zeros((h, w), dtype=np.float32)
    for scale, sigma in scales_sigmas:
        sh, sw = max(1, h // scale), max(1, w // scale)
        n = np.random.normal(0, sigma, (sh, sw)).astype(np.float32)
        total += cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    return total

def generate_ground_texture(texture_type, size_px, config=None):
    h = w = size_px

    if texture_type == "wood":
        base = np.array([30, 50, 80], dtype=np.float32)
        tex = np.tile(base, (h, w, 1))
        plank_w = 60
        for x in range(0, w, plank_w):
            factor = random.uniform(0.9, 1.1)
            end_x = min(x + plank_w, w)
            tex[:, x:end_x] = np.clip(base * factor, 0, 255)
            if end_x < w:
                tex[:, end_x - 1:end_x] = np.clip(base * 0.7, 0, 255)
        tex += _multiscale_noise(h, w, [(4, 6.0), (40, 3.0)])[..., None]

    elif texture_type == "gym_floor":
        cfg = config or {}
        base = np.array([70, 120, 165], dtype=np.float32)
        tex = np.tile(base, (h, w, 1))
        plank_w = 45
        for x in range(0, w, plank_w):
            factor = random.uniform(0.94, 1.06)
            end_x = min(x + plank_w, w)
            tex[:, x:end_x] = np.clip(base * factor, 0, 255)
        tex += _multiscale_noise(h, w, [(4, 4.0), (50, 3.0)])[..., None]
        tex = np.clip(tex, 0, 255).astype(np.uint8)

        line_colors = cfg.get("gym_line_colors_bgr", {
            "white": (235, 235, 235), "yellow": (40, 210, 235),
            "red": (55, 55, 195), "blue": (195, 110, 50),
        })
        lw_lo, lw_hi = cfg.get("gym_court_line_width_px_range", (4, 9))

        def line_w():
            return random.randint(lw_lo, lw_hi)

        def rand_color():
            return random.choice(list(line_colors.values()))

        margin_px = int(0.08 * size_px)
        p1 = (margin_px, margin_px)
        p2 = (size_px - margin_px, size_px - margin_px)
        cv2.rectangle(tex, p1, p2, rand_color(), line_w())
        cv2.line(tex, (margin_px, size_px // 2), (size_px - margin_px, size_px // 2), rand_color(), line_w())
        cv2.line(tex, (size_px // 2, margin_px), (size_px // 2, size_px - margin_px), rand_color(), line_w())

        cv2.circle(tex, (size_px // 2, size_px // 2), int(0.10 * size_px), rand_color(), line_w())
        for cy_c in (margin_px + int(0.18 * size_px), size_px - margin_px - int(0.18 * size_px)):
            cv2.circle(tex, (size_px // 2, cy_c), int(0.07 * size_px), rand_color(), max(2, line_w() - 2))

        key_w, key_h = int(0.16 * size_px), int(0.24 * size_px)
        cv2.rectangle(tex, (size_px // 2 - key_w // 2, margin_px),
                      (size_px // 2 + key_w // 2, margin_px + key_h), rand_color(), max(2, line_w() - 1))
        cv2.rectangle(tex, (size_px // 2 - key_w // 2, size_px - margin_px - key_h),
                      (size_px // 2 + key_w // 2, size_px - margin_px), rand_color(), max(2, line_w() - 1))

        if random.random() < cfg.get("gym_logo_prob", 0.6):
            logo_color = rand_color()
            logo_r = int(0.05 * size_px)
            cv2.circle(tex, (size_px // 2, size_px // 2), logo_r, logo_color, -1)
            cv2.circle(tex, (size_px // 2, size_px // 2), int(logo_r * 0.55), tuple(int(c) for c in base), -1)

        if random.random() < cfg.get("gym_number_prob", 0.5):
            for _ in range(random.randint(1, 3)):
                num = str(random.randint(0, 99))
                pos = (random.randint(margin_px, size_px - margin_px - 60),
                       random.randint(margin_px + 60, size_px - margin_px))
                cv2.putText(tex, num, pos, cv2.FONT_HERSHEY_SIMPLEX,
                            random.uniform(1.5, 3.0), rand_color(), random.randint(3, 6), cv2.LINE_AA)

        tex = tex.astype(np.float32)
        tex += _multiscale_noise(h, w, [(3, 2.5)])[..., None]

    elif texture_type == "grass":
        base = np.array([35, 90, 35], dtype=np.float32)
        tex = np.tile(base, (h, w, 1))
        blades = _multiscale_noise(h, w, [(2, 18.0), (15, 10.0), (60, 6.0)])[..., None]
        tex += blades * np.array([0.4, 1.0, 0.4], dtype=np.float32)

    elif texture_type == "asphalt":
        base = np.array([55, 55, 58], dtype=np.float32)
        tex = np.tile(base, (h, w, 1))
        tex += _multiscale_noise(h, w, [(2, 10.0), (30, 4.0)])[..., None]

    elif texture_type == "concrete":
        base = np.array([150, 150, 148], dtype=np.float32)
        tex = np.tile(base, (h, w, 1))
        tex += _multiscale_noise(h, w, [(3, 8.0), (50, 6.0)])[..., None]

    elif texture_type == "tile":
        base = np.array([140, 138, 130], dtype=np.float32)
        tex = np.tile(base, (h, w, 1))
        grout = np.clip(base * 0.55, 0, 255)
        tile_px = 80
        for x in range(0, w, tile_px):
            tex[:, max(x - 1, 0):x + 1] = grout
        for y in range(0, h, tile_px):
            tex[max(y - 1, 0):y + 1, :] = grout
        tex += _multiscale_noise(h, w, [(5, 4.0)])[..., None]

    elif texture_type == "dirt":
        base = np.array([45, 75, 110], dtype=np.float32)
        tex = np.tile(base, (h, w, 1))
        tex += _multiscale_noise(h, w, [(2, 14.0), (20, 10.0), (70, 6.0)])[..., None]

    else:
        tex = np.tile(np.array([80, 80, 80], dtype=np.float32), (h, w, 1))

    tex = np.clip(tex, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(tex, (3, 3), 0)

def compute_ground_homography(R, altitude, drone_x, drone_y, fx, fy, cx, cy):
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    A = np.array([
        [1.0, 0.0, -drone_x],
        [0.0, 1.0, -drone_y],
        [0.0, 0.0, altitude],
    ], dtype=np.float64)
    return K @ (R @ A)

def warp_ground_texture(texture, ground_homography, texels_per_meter, width, height):
    tex_h, tex_w = texture.shape[:2]
    tex_cx, tex_cy = tex_w / 2.0, tex_h / 2.0
    tpm = float(texels_per_meter)
    T = np.array([
        [1.0 / tpm, 0.0, -tex_cx / tpm],
        [0.0, 1.0 / tpm, -tex_cy / tpm],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    M = ground_homography @ T
    return cv2.warpPerspective(
        texture, M, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )

def get_camera_intrinsics(width, height):
    h_fov_rad = np.radians(CONFIG["camera_h_fov"])
    v_fov_rad = np.radians(CONFIG["camera_v_fov"])
    
    fx = width / (2.0 * np.tan(h_fov_rad / 2.0))
    fy = height / (2.0 * np.tan(v_fov_rad / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    
    return fx, fy, cx, cy

def compute_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def project_points(pts_w, R, fx, fy, cx, cy):
    pts_c = (R @ pts_w.T).T
    z = np.clip(pts_c[:, 2], 0.05, None)
    u = fx * (pts_c[:, 0] / z) + cx
    v = fy * (pts_c[:, 1] / z) + cy
    return np.stack([u, v], axis=1).astype(np.float32)

def apply_drone_rotation(marker_rgba, width, height, altitude, roll_deg, pitch_deg, yaw_deg,
                          drone_x, drone_y, sun_az_deg, sun_elev_deg, config):
    fx, fy, cx, cy = get_camera_intrinsics(width, height)
    R = compute_rotation_matrix(roll_deg, pitch_deg, yaw_deg)

    s = config["marker_real_size"]
    base_xy = np.array([[-s/2, -s/2], [s/2, -s/2], [s/2, s/2], [-s/2, s/2]])
    pts_w = np.column_stack([
        base_xy[:, 0] - drone_x,
        base_xy[:, 1] - drone_y,
        np.full(4, altitude)
    ])
    pts_img = project_points(pts_w, R, fx, fy, cx, cy)

    shadow_len = config["shadow_length_coeff"] * altitude / max(np.tan(np.radians(sun_elev_deg)), 0.2)
    shadow_dx = shadow_len * np.cos(np.radians(sun_az_deg))
    shadow_dy = shadow_len * np.sin(np.radians(sun_az_deg))
    pts_w_shadow = pts_w + np.array([shadow_dx, shadow_dy, 0.0])
    pts_shadow_img = project_points(pts_w_shadow, R, fx, fy, cx, cy)

    h_src, w_src = marker_rgba.shape[:2]
    pts_src = np.array([[0, 0], [w_src-1, 0], [w_src-1, h_src-1], [0, h_src-1]], dtype=np.float32)

    H_marker = cv2.getPerspectiveTransform(pts_src, pts_img)
    marker_warped = cv2.warpPerspective(
        marker_rgba, H_marker, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
    )

    shadow_warped = add_shadow(pts_src, pts_shadow_img, width, height, altitude, config)

    ground_homography = compute_ground_homography(R, altitude, drone_x, drone_y, fx, fy, cx, cy)

    return marker_warped, shadow_warped, pts_img, ground_homography

def add_shadow(pts_src, pts_shadow_img, width, height, altitude, config):
    shadow_src = np.zeros((500, 500, 4), dtype=np.uint8)
    shadow_src[:, :, 3] = 130

    H_shadow = cv2.getPerspectiveTransform(pts_src, pts_shadow_img)
    shadow_warped = cv2.warpPerspective(
        shadow_src, H_shadow, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
    )

    blur_size = int(config["shadow_blur_base_px"] + altitude * config["shadow_blur_altitude_coeff"])
    blur_size = max(3, blur_size)
    if blur_size % 2 == 0:
        blur_size += 1

    return cv2.GaussianBlur(shadow_warped, (blur_size, blur_size), 0)

def calculate_marker_size(pts_marker_img):
    d1 = np.linalg.norm(pts_marker_img[0] - pts_marker_img[1])
    d2 = np.linalg.norm(pts_marker_img[1] - pts_marker_img[2])
    d3 = np.linalg.norm(pts_marker_img[2] - pts_marker_img[3])
    d4 = np.linalg.norm(pts_marker_img[3] - pts_marker_img[0])
    return int(np.mean([d1, d2, d3, d4]))

def apply_motion_blur(image, length, angle_deg):
    if length <= 1:
        return image
    
    size = int(max(length, 3))
    if size % 2 == 0:
        size += 1
        
    kernel = np.zeros((size, size))
    center = size // 2
    
    angle_rad = np.radians(angle_deg)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    for i in range(size):
        offset = i - center
        x = int(round(center + offset * dx))
        y = int(round(center + offset * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
            
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    else:
        return image
        
    return cv2.filter2D(image, -1, kernel)

def apply_rolling_shutter(image, shift_max_px):
    if abs(shift_max_px) < 1:
        return image
        
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    
    shift_profile = (np.arange(h) / (h - 1)) * shift_max_px
    map_x = map_x + shift_profile[:, np.newaxis]
    
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def apply_lens_distortion(image, k1, k2=0.0, k3=0.0, p1=0.0, p2=0.0):
    h, w = image.shape[:2]
    f = max(h, w)
    cx, cy = w / 2.0, h / 2.0

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    x = (grid_x - cx) / f
    y = (grid_y - cy) / f
    r2 = x ** 2 + y ** 2
    r4 = r2 ** 2
    r6 = r2 * r4

    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_tan = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    y_tan = p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

    map_x = ((x * radial + x_tan) * f + cx).astype(np.float32)
    map_y = ((y * radial + y_tan) * f + cy).astype(np.float32)

    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def generate_hot_pixel_mask(width, height, count_range):
    mask = np.zeros((height, width), dtype=bool)
    n = random.randint(*count_range)
    for _ in range(n):
        yx = (random.randint(0, height - 1), random.randint(0, width - 1))
        mask[yx] = True
    return mask

def apply_sensor_noise(image, config, hot_pixel_mask=None):
    luma_sigma_range = config["noise_luma_sigma_range"]
    chroma_sigma_range = config["noise_chroma_sigma_range"]
    lowlight_boost = config["chroma_lowlight_boost"]
    shot_coeff = config["shot_noise_coeff"]

    luma_sigma = random.uniform(*luma_sigma_range)
    chroma_sigma = random.uniform(*chroma_sigma_range)

    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    luma = ycc[:, :, 0]
    darkness = 1.0 - luma / 255.0
    lowlight_factor = 1.0 + darkness * lowlight_boost

    shot_sigma = np.sqrt(np.clip(luma, 1.0, 255.0)) * shot_coeff

    ycc[:, :, 0] += np.random.normal(0, 1.0, luma.shape) * (luma_sigma + shot_sigma * 0.3)
    ycc[:, :, 1] += np.random.normal(0, chroma_sigma, luma.shape) * lowlight_factor
    ycc[:, :, 2] += np.random.normal(0, chroma_sigma, luma.shape) * lowlight_factor

    ycc = np.clip(ycc, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    if hot_pixel_mask is not None and hot_pixel_mask.any():
        out[hot_pixel_mask] = config["hot_pixel_value"]

    return out

def apply_vignette(image, strength):
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    ccx, ccy = w / 2.0, h / 2.0
    max_r = np.sqrt(ccx ** 2 + ccy ** 2)
    r = np.sqrt((xx - ccx) ** 2 + (yy - ccy) ** 2) / max_r
    mask = np.clip(1.0 - strength * (r ** 2), 0.0, 1.0).astype(np.float32)
    out = image.astype(np.float32) * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_auto_exposure(image, gain):
    out = image.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_focus_blur(image, ksize):
    ksize = int(round(ksize))
    if ksize < 3:
        return image
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_specular_highlights(image, config):
    h, w = image.shape[:2]
    n = random.randint(*config["specular_highlight_count_range"])
    if n == 0:
        return image

    overlay = np.zeros((h, w), dtype=np.float32)
    diag = np.sqrt(h ** 2 + w ** 2)
    for _ in range(n):
        length = diag * random.uniform(*config["specular_highlight_length_frac_range"])
        width_streak = random.uniform(8, 30)
        angle = random.uniform(0, 180)
        cx = random.uniform(0, w)
        cy = random.uniform(0, h)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ang_rad = np.radians(angle)
        u = (xx - cx) * np.cos(ang_rad) + (yy - cy) * np.sin(ang_rad)
        v = -(xx - cx) * np.sin(ang_rad) + (yy - cy) * np.cos(ang_rad)
        blob = np.exp(-(u ** 2) / (2 * (length / 2.5) ** 2) - (v ** 2) / (2 * width_streak ** 2))
        overlay += blob * random.uniform(*config["specular_highlight_intensity_range"])

    out = image.astype(np.float32) + overlay[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_neon_and_white_balance(image, t_abs, config, wb_temp, wb_green):
    freq = config["neon_flicker_freq_hz"]
    flicker_amp = random.uniform(*config["neon_flicker_amplitude_range"])
    flicker_gain = 1.0 + flicker_amp * math.sin(2 * math.pi * freq * t_abs + random.uniform(0, 2 * math.pi) * 0.05)

    green_tint = random.uniform(*config["neon_green_tint_range"])

    r_gain = 1.0 + 0.20 * wb_temp
    b_gain = 1.0 - 0.20 * wb_temp
    g_gain = 1.0 + 0.12 * wb_green + green_tint

    out = image.astype(np.float32) * flicker_gain
    out[:, :, 0] *= b_gain   # B
    out[:, :, 1] *= g_gain   # G
    out[:, :, 2] *= r_gain   # R
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_vibration_jitter(image, amplitude_px):
    if amplitude_px <= 0:
        return image
    dx = np.random.normal(0, amplitude_px)
    dy = np.random.normal(0, amplitude_px)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def apply_jpeg_compression(image, quality_choices):
    quality = random.choice(quality_choices)
    ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def apply_marker_imperfections(marker_rgba, config):
    h, w = marker_rgba.shape[:2]
    out = marker_rgba.astype(np.float32).copy()

    black_level = random.uniform(*config["marker_black_level_range"])
    white_level = random.uniform(*config["marker_white_level_range"])
    normalized = out[:, :, :3] / 255.0
    out[:, :, :3] = black_level + normalized * (white_level - black_level)

    paper_noise = np.random.normal(0, config["marker_paper_noise_sigma"], (h, w, 1))
    out[:, :, :3] = np.clip(out[:, :, :3] + paper_noise, 0, 255)

    if random.random() < config["marker_warp_prob"]:
        amp = random.uniform(*config["marker_warp_amplitude_px_range"])
        freq = random.uniform(1.0, 2.0)
        phase = random.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        disp_x = amp * np.sin(2 * np.pi * freq * yy / h + phase)
        disp_y = amp * np.cos(2 * np.pi * freq * xx / w + phase)
        map_x = (xx + disp_x).astype(np.float32)
        map_y = (yy + disp_y).astype(np.float32)
        out = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    if random.random() < config["marker_corner_lift_prob"]:
        corner = random.choice([(0, 0), (0, w), (h, 0), (h, w)])
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((yy - corner[0]) ** 2 + (xx - corner[1]) ** 2)
        radius = min(h, w) * random.uniform(0.2, 0.4)
        falloff = np.clip(1.0 - dist / radius, 0, 1) * random.uniform(0.3, 0.6)
        out[:, :, :3] *= (1.0 - falloff[..., None])

    return np.clip(out, 0, 255).astype(np.uint8)

def composite_images(floor, shadow, marker):
    floor_f = floor.astype(np.float32)
    
    shadow_rgb = shadow[:, :, :3].astype(np.float32)
    shadow_alpha = np.expand_dims(shadow[:, :, 3].astype(np.float32) / 255.0, axis=2)
    bg_with_shadow = shadow_rgb * shadow_alpha + floor_f * (1.0 - shadow_alpha)
    
    marker_rgb = marker[:, :, :3].astype(np.float32)
    marker_alpha = np.expand_dims(marker[:, :, 3].astype(np.float32) / 255.0, axis=2)
    final = marker_rgb * marker_alpha + bg_with_shadow * (1.0 - marker_alpha)
    
    return np.clip(final, 0, 255).astype(np.uint8)

def compute_marker_visibility(pts_img, width, height):
    xs, ys = pts_img[:, 0], pts_img[:, 1]
    raw_xmin, raw_xmax = xs.min(), xs.max()
    raw_ymin, raw_ymax = ys.min(), ys.max()
    raw_area = max(raw_xmax - raw_xmin, 1e-6) * max(raw_ymax - raw_ymin, 1e-6)

    clip_xmin = max(0.0, raw_xmin)
    clip_xmax = min(float(width), raw_xmax)
    clip_ymin = max(0.0, raw_ymin)
    clip_ymax = min(float(height), raw_ymax)

    visible_w = max(0.0, clip_xmax - clip_xmin)
    visible_h = max(0.0, clip_ymax - clip_ymin)
    visible_area = visible_w * visible_h
    fraction = visible_area / raw_area if raw_area > 0 else 0.0

    if visible_w <= 0 or visible_h <= 0:
        return None, 0.0, 0.0, 0.0

    return (int(clip_xmin), int(clip_xmax), int(clip_ymin), int(clip_ymax)), fraction, visible_w, visible_h

def render_frame(marker_base, ground_texture, width, height, fx,
                  altitude, roll, pitch, yaw, yaw_rate, speed, drone_x, drone_y,
                  sun_az_deg, sun_elev_deg, focus_distance_m, autofocus_hunting,
                  k1, k2, k3, p1, p2, wb_temp, wb_green, t_abs, hot_pixel_mask, config):
    marker_w, shadow_w, pts_img, ground_H = apply_drone_rotation(
        marker_base, width, height, altitude, roll, pitch, yaw,
        drone_x, drone_y, sun_az_deg, sun_elev_deg, config
    )

    scene = warp_ground_texture(ground_texture, ground_H, config["ground_texels_per_meter"], width, height)
    scene = composite_images(scene, shadow_w, marker_w)

    scene = apply_specular_highlights(scene, config)

    vib_amp = random.uniform(*config["vibration_amplitude_px_range"])
    scene = apply_vibration_jitter(scene, vib_amp)

    motion_dist_m = speed * config["exposure_time"]
    motion_blur_len = motion_dist_m * (fx / altitude) + vib_amp * config["vibration_blur_coeff"]
    motion_angle = 90.0 + yaw_rate * config["exposure_time"] * 10.0 + random.uniform(-3, 3)
    scene = apply_motion_blur(scene, motion_blur_len, motion_angle)

    lateral_speed = speed * np.sin(np.radians(yaw))
    shutter_shift_m = lateral_speed * config["rolling_shutter_readout"]
    shutter_shift_px = shutter_shift_m * (fx / altitude)
    scene = apply_rolling_shutter(scene, shutter_shift_px)

    scene = apply_lens_distortion(scene, k1, k2, k3, p1, p2)

    defocus_m = abs(altitude - focus_distance_m)
    ksize = config["dof_blur_base_px"] + defocus_m * config["dof_blur_coeff_px_per_m"]
    if autofocus_hunting:
        ksize += config["autofocus_hunt_extra_ksize"]
    ksize = min(ksize, config["dof_max_ksize"])
    scene = apply_focus_blur(scene, ksize)

    scene = apply_neon_and_white_balance(scene, t_abs, config, wb_temp, wb_green)

    vignette_strength = random.uniform(*config["vignette_strength_range"])
    scene = apply_vignette(scene, vignette_strength)

    ae_gain = random.uniform(*config["ae_gain_range"])
    scene = apply_auto_exposure(scene, ae_gain)

    scene = apply_sensor_noise(scene, config, hot_pixel_mask=hot_pixel_mask)
    scene = apply_jpeg_compression(scene, config["jpeg_quality_choices"])

    meta = {
        "vibration_px": vib_amp,
        "vignette_strength": vignette_strength,
        "ae_gain": ae_gain,
    }
    return scene, pts_img, ground_H, meta

def draw_vibration_shift(amplitude_px: float) -> Tuple[float, float]:
    """Tire le meme (dx,dy) que apply_vibration_jitter() aurait tire en
    interne, mais en le RETOURNANT : la fonction d'origine ne renvoie que
    l'image decalee, pas le decalage lui-meme, ce qui empechait de corriger
    la verite-terrain en consequence. Reproduit son tirage exactement
    (meme loi, meme ordre de tirage dx puis dy) pour qu'appeler cette
    fonction PUIS warpAffine([[1,0,dx],[0,1,dy]]) soit rigoureusement
    equivalent a apply_vibration_jitter(image, amplitude_px)."""
    if amplitude_px <= 0:
        return 0.0, 0.0
    return float(np.random.normal(0, amplitude_px)), float(np.random.normal(0, amplitude_px))


def rolling_shutter_shift_x(y_px, shift_max_px: float, height: int):
    """Deplacement horizontal qu'apply_rolling_shutter() applique a un point
    situe (apres tout decalage anterieur) sur la ligne y_px — forme fermee,
    verifiee empiriquement (cv2.remap avec map_x=x+profil, map_y=y) : le
    point source apparait dans l'image de sortie a x_out = x_src - profil(y).
    Pas d'iteration necessaire (contrairement a la distorsion optique) : le
    profil ne depend que de y, qui n'est pas lui-meme modifie par cette
    etape."""
    if abs(shift_max_px) < 1:
        return np.zeros_like(np.asarray(y_px, dtype=np.float64))
    return (np.asarray(y_px, dtype=np.float64) / (height - 1)) * shift_max_px


def undistort_points_to_output(pts_px, width, height, k1, k2, k3, p1, p2, iters=8):
    """Position, dans l'image FINALE (apres apply_lens_distortion), d'un point
    donne dans l'image AVANT distorsion (ce que rend project_points()).

    BUG CORRIGE PAR RAPPORT AU SCRIPT D'ORIGINE : process_verification_marker
    enregistre pts_img (calcule par project_points, donc AVANT distorsion)
    comme verite-terrain "aruco_corners", alors que render_frame() applique
    apply_lens_distortion() a l'image ENSUITE. Pour k1 ~ -0.13 (borne haute
    de la plage configuree), l'ecart mesure au coin d'un marqueur atteint
    ~20 px a 640x480 — largement de quoi faire rater un appariement IoU, ou
    pire, faire apprendre un mauvais crop a un reseau entraine sur ce JSON.

    apply_lens_distortion(image, ...) construit, pour chaque pixel de
    SORTIE, la coordonnee d'ENTREE a echantillonner :
        src_norm = distort(out_norm), avec distort(x) = x*radial(x) + tangentiel(x)
    Pour replacer un point donne dans l'image d'ENTREE (non distordue) au
    bon endroit dans l'image de SORTIE, il faut donc l'operation inverse :
    trouver out_norm tel que distort(out_norm) = src_norm_cible. distort()
    n'a pas d'inverse analytique ; a la difference de cv2.undistortPoints
    (qui suppose src_norm connu et cherche out_norm, exactement notre cas),
    on la resout par point fixe — quelques iterations suffisent car la
    distorsion reste faible (|k1| < 0,15) :
        out_norm <- out_norm + (src_norm_cible - distort(out_norm))
    """
    f = float(max(width, height))
    cx, cy = width / 2.0, height / 2.0
    target = (np.asarray(pts_px, dtype=np.float64) - [cx, cy]) / f
    out = target.copy()
    for _ in range(iters):
        r2 = out[..., 0] ** 2 + out[..., 1] ** 2
        r4 = r2 ** 2
        r6 = r2 * r4
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        x_tan = 2 * p1 * out[..., 0] * out[..., 1] + p2 * (r2 + 2 * out[..., 0] ** 2)
        y_tan = p1 * (r2 + 2 * out[..., 1] ** 2) + 2 * p2 * out[..., 0] * out[..., 1]
        src = np.stack([out[..., 0] * radial + x_tan, out[..., 1] * radial + y_tan], axis=-1)
        out = out + (target - src)
    return (out * f + [cx, cy]).astype(np.float32)


# ==============================================================================
# Génération du marqueur (remplace load_marker/cairosvg — DICT_4X4_1000,
# cohérent avec le nommage 4x4_1000-N.svg de l'original)
# ==============================================================================
_DICT_NAME = os.environ.get("ARUCO_DICT_NAME", "DICT_4X4_1000")
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, _DICT_NAME))


def generate_marker_rgba(marker_id: int, size: int = 600) -> np.ndarray:
    """Marqueur RGBA opaque (alpha=255 partout : sticker physique imprimé,
    pas de zone transparente), bordure noire incluse (borderBits=1) — même
    convention que les SVG 4x4_1000-N.svg de l'original."""
    gray = cv2.aruco.generateImageMarker(_ARUCO_DICT, int(marker_id), size, borderBits=1)
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, 0] = gray
    rgba[:, :, 1] = gray
    rgba[:, :, 2] = gray
    rgba[:, :, 3] = 255
    return rgba


# ==============================================================================
# Scène indépendante multi-marqueurs (remplace generate_cnn_examples_for_marker)
# ==============================================================================
EXTRA_CONFIG = {
    # Nb de marqueurs par scene (bornes incluses) ; chacun avec son propre
    # tirage de pose (altitude/roll/pitch/yaw independants, comme dans
    # generate_cnn_examples_for_marker), sur le MEME sol/eclairage/bruit —
    # une seule scene = une seule prise de vue.
    "markers_per_scene_range": (1, 3),
    # Distance minimale entre les centres de deux marqueurs de la même scène,
    # en fraction du rayon d'empreinte au sol : évite deux marqueurs
    # totalement superposés (l'original ne gère qu'un marqueur par image et
    # n'a donc pas ce probleme).
    "scene_min_center_sep_frac": 0.35,
}


def _place_marker(config, R, altitude, fx, fy, cx, cy, width, height,
                  footprint_radius_m, existing_centers_m, min_sep_m):
    """Tire une position au sol pour UN marqueur, sous une pose CAMERA deja
    fixee (R, altitude) — partagee par toute la scene, voir la note dans
    render_independent_scene(). Respecte la visibilite minimale et l'ecart
    minimal aux marqueurs deja places.

    BUG CORRIGE PAR RAPPORT A UNE PREMIERE VERSION DE CE SCRIPT : cette
    fonction tirait auparavant sa PROPRE altitude/roll/pitch/yaw pour
    chaque marqueur — physiquement faux dans une scene multi-marqueurs
    (plusieurs marqueurs vus par UNE camera a UN instant partagent
    forcement la meme attitude). Le symptome etait visible sur les
    coordonnees : le sol et les marqueurs autres que le dernier place
    n'etaient pas mutuellement coherents en perspective, et comparer un
    marqueur detecte a sa verite-terrain donnait un IoU degrade
    (~0,2-0,3 au lieu de >0,6) sans lien avec un vrai probleme de
    detection.
    """
    s = config["marker_real_size"]
    base_xy = np.array([[-s/2, -s/2], [s/2, -s/2], [s/2, s/2], [-s/2, s/2]])
    for _attempt in range(config["cnn_placement_max_attempts"]):
        place_radius = footprint_radius_m * random.uniform(0.0, 1.15)
        place_angle = random.uniform(0.0, 2 * math.pi)
        cand_x = -place_radius * math.cos(place_angle)
        cand_y = -place_radius * math.sin(place_angle)

        if any(math.hypot(cand_x - ex, cand_y - ey) < min_sep_m
               for ex, ey in existing_centers_m):
            continue

        pts_w = np.column_stack([base_xy[:, 0] - cand_x, base_xy[:, 1] - cand_y,
                                 np.full(4, altitude)])
        pts_img = project_points(pts_w, R, fx, fy, cx, cy)
        clipped, vis_fraction, vis_w, vis_h = compute_marker_visibility(pts_img, width, height)

        if (clipped is not None and vis_fraction >= config["cnn_min_visible_fraction"]
                and vis_w >= config["cnn_min_visible_px"]
                and vis_h >= config["cnn_min_visible_px"]):
            return dict(x=cand_x, y=cand_y, pts_img=pts_img,
                        vis_fraction=vis_fraction, vis_w=vis_w, vis_h=vis_h,
                        clipped=clipped)
    return None


def render_independent_scene(marker_ids, config, extra):
    """Une scene = un sol + un eclairage + un bruit, PLUSIEURS marqueurs
    (chacun avec sa propre pose), rendue PLEIN CADRE (pas de crop CNN).

    Reprend telles quelles les fonctions physiques de l'original
    (generate_ground_texture, apply_drone_rotation, render_frame, ...) ;
    seule la boucle d'orchestration change pour gerer N marqueurs sur une
    scene au lieu d'un crop centre sur un seul.

    Retourne (image_bgr, [verite_terrain...], meta_scene).
    """
    width, height = config["output_resolution"]
    fx, fy, cx, cy = get_camera_intrinsics(width, height)
    half_diag_px = math.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2)

    texture_type = random.choices(
        config["ground_texture_types"], weights=config.get("ground_texture_weights"), k=1
    )[0]
    tex_size_px = int(config["ground_texture_size_m"] * config["ground_texels_per_meter"])
    ground_texture = generate_ground_texture(texture_type, tex_size_px, config)

    sun_az_deg = random.uniform(0.0, 360.0)
    sun_elev_deg = random.uniform(*config["sun_elevation_range_deg"])
    hot_pixel_mask = generate_hot_pixel_mask(width, height, config["hot_pixel_count_range"])
    k1 = random.uniform(*config["k1_distortion_range"])
    k2 = random.uniform(*config["k2_distortion_range"])
    k3 = random.uniform(*config["k3_distortion_range"])
    p1 = random.uniform(*config["p1_distortion_range"])
    p2 = random.uniform(*config["p2_distortion_range"])
    wb_temp = random.uniform(*config["wb_temp_range"])
    wb_green = random.uniform(*config["wb_green_range"])
    t_abs = random.uniform(0.0, 10.0)

    # Une seule pose CAMERA pour toute la scene (altitude, roll, pitch, yaw) :
    # tous les marqueurs de la scene sont vus par la meme camera au meme
    # instant, seule leur position au sol differe (voir la note dans
    # _place_marker). alt_ref sert aussi de reference pour le flou de
    # defocalisation et le flou de mouvement de toute la scene.
    alt_ref = random.uniform(config["altitude_min"], config["altitude_max"])
    roll_ref = random.uniform(-config["roll_max_deg"], config["roll_max_deg"])
    pitch_ref = random.uniform(-config["pitch_max_deg"], config["pitch_max_deg"])
    yaw_ref = random.uniform(0.0, 360.0)
    R_ref = compute_rotation_matrix(roll_ref, pitch_ref, yaw_ref)

    footprint_radius_m = (half_diag_px / fx) * alt_ref
    focus_distance_m = float(np.clip(
        alt_ref + random.uniform(*config["focus_error_range_m"]),
        config["altitude_min"], config["altitude_max"]))
    autofocus_hunting = random.random() < config["autofocus_hunt_event_prob"]

    placements = []
    centers_m = []
    min_sep_m = footprint_radius_m * extra["scene_min_center_sep_frac"]
    for mid in marker_ids:
        p = _place_marker(config, R_ref, alt_ref, fx, fy, cx, cy, width, height,
                          footprint_radius_m, centers_m, min_sep_m)
        if p is None:
            continue
        p["marker_id"] = mid
        p["altitude"], p["roll"], p["pitch"], p["yaw"] = alt_ref, roll_ref, pitch_ref, yaw_ref
        placements.append(p)
        centers_m.append((p["x"], p["y"]))
    if not placements:
        return None, [], {}

    # --- composition : sol -> ombre+marqueur pour chaque placement, dans
    # l'ordre (celui qui est place en dernier peut recouvrir les precedents,
    # comme deux marqueurs poses proches l'un de l'autre en vrai).
    scene = None
    for p in placements:
        marker_base_clean = generate_marker_rgba(p["marker_id"], size=600)
        marker_base = apply_marker_imperfections(marker_base_clean, config)
        marker_w, shadow_w, pts_img, ground_H = apply_drone_rotation(
            marker_base, width, height, p["altitude"], p["roll"], p["pitch"], p["yaw"],
            p["x"], p["y"], sun_az_deg, sun_elev_deg, config)
        p["pts_img"] = pts_img              # recalcule avec la vraie geometrie de rendu
        if scene is None:
            scene = warp_ground_texture(ground_texture, ground_H,
                                        config["ground_texels_per_meter"], width, height)
        scene = composite_images(scene, shadow_w, marker_w)

    # --- degradations globales de la scene (une seule fois, comme une vraie
    # prise de vue), en reutilisant EXACTEMENT le corps de render_frame() a
    # partir de la composition — dupliquee ici car render_frame() compose
    # elle-meme un seul marqueur ; on lui evite donc de recomposer.
    speed = max(0.0, config["drone_speed"] + random.gauss(0.0, 1.0))
    yaw_rate = random.gauss(0.0, 8.0)

    scene = apply_specular_highlights(scene, config)

    vib_amp = random.uniform(*config["vibration_amplitude_px_range"])
    # dx,dy tires ICI (pas dans apply_vibration_jitter) pour pouvoir corriger
    # la verite-terrain du meme decalage — voir draw_vibration_shift().
    vib_dx, vib_dy = draw_vibration_shift(vib_amp)
    if vib_dx or vib_dy:
        M_vib = np.float32([[1, 0, vib_dx], [0, 1, vib_dy]])
        scene = cv2.warpAffine(scene, M_vib, (scene.shape[1], scene.shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    motion_dist_m = speed * config["exposure_time"]
    motion_blur_len = motion_dist_m * (fx / alt_ref) + vib_amp * config["vibration_blur_coeff"]
    motion_angle = 90.0 + yaw_rate * config["exposure_time"] * 10.0 + random.uniform(-3, 3)
    scene = apply_motion_blur(scene, motion_blur_len, motion_angle)

    lateral_speed = speed * math.sin(math.radians(placements[0]["yaw"]))
    shutter_shift_m = lateral_speed * config["rolling_shutter_readout"]
    shutter_shift_px = shutter_shift_m * (fx / alt_ref)
    scene = apply_rolling_shutter(scene, shutter_shift_px)

    scene = apply_lens_distortion(scene, k1, k2, k3, p1, p2)

    defocus_m = abs(alt_ref - focus_distance_m)
    ksize = config["dof_blur_base_px"] + defocus_m * config["dof_blur_coeff_px_per_m"]
    if autofocus_hunting:
        ksize += config["autofocus_hunt_extra_ksize"]
    ksize = min(ksize, config["dof_max_ksize"])
    scene = apply_focus_blur(scene, ksize)

    scene = apply_neon_and_white_balance(scene, t_abs, config, wb_temp, wb_green)

    vignette_strength = random.uniform(*config["vignette_strength_range"])
    scene = apply_vignette(scene, vignette_strength)

    ae_gain = random.uniform(*config["ae_gain_range"])
    scene = apply_auto_exposure(scene, ae_gain)

    scene = apply_sensor_noise(scene, config, hot_pixel_mask=hot_pixel_mask)
    scene = apply_jpeg_compression(scene, config["jpeg_quality_choices"])

    # Verite-terrain corrigee des TROIS warps geometriques poses sur l'image
    # apres le rendu des coins (pts_img), dans l'ORDRE EXACT du pipeline
    # ci-dessus : vibration (translation constante) -> rolling shutter
    # (cisaillement dependant de y) -> distorsion optique (non lineaire).
    # Aucun n'etait corrige dans le script d'origine ; mesure sur un cas
    # reel du dataset : ~13-20 px d'ecart residuel, dont l'essentiel venait
    # du rolling shutter (jusqu'a ~25 px a 8 m/s pleine vitesse laterale,
    # altitude basse), le reste de la distorsion (~5-20 px selon k1 et la
    # position dans le cadre) et un peu de la vibration (<3 px, 1 sigma).
    ground_truth = []
    for p in placements:
        pts = p["pts_img"] + np.float32([vib_dx, vib_dy])
        pts[:, 0] -= rolling_shutter_shift_x(pts[:, 1], shutter_shift_px, height)
        corners_final = undistort_points_to_output(pts, width, height, k1, k2, k3, p1, p2)
        ground_truth.append({
            "marker_id": int(p["marker_id"]),
            "corners_px": corners_final.tolist(),
            "corners_px_pre_warp": p["pts_img"].tolist(),
            "visible_fraction": round(float(p["vis_fraction"]), 3),
            "visible_px": [round(float(p["vis_w"]), 1), round(float(p["vis_h"]), 1)],
            "altitude_m": round(float(p["altitude"]), 2),
            "roll_deg": round(float(p["roll"]), 1),
            "pitch_deg": round(float(p["pitch"]), 1),
            "yaw_deg": round(float(p["yaw"]), 1),
            "marker_side_px": calculate_marker_size(corners_final),
        })

    meta = {
        "ground_texture": texture_type,
        "sun_elevation_deg": round(sun_elev_deg, 1),
        "sun_azimuth_deg": round(sun_az_deg, 1),
        "focus_distance_m": round(focus_distance_m, 2),
        "altitude_ref_m": round(alt_ref, 2),
        "autofocus_hunting": bool(autofocus_hunting),
        "motion_blur_len_px": round(float(motion_blur_len), 2),
        "defocus_blur_ksize": round(float(ksize), 1),
        "vignette_strength": round(float(vignette_strength), 3),
        "ae_gain": round(float(ae_gain), 3),
        "vibration_px": round(float(vib_amp), 3),
        "vibration_shift_xy_px": [round(vib_dx, 2), round(vib_dy, 2)],
        "rolling_shutter_shift_max_px": round(float(shutter_shift_px), 2),
        "wb_temp": round(float(wb_temp), 3),
        "lens_distortion_k1": round(float(k1), 4),
        "n_markers": len(placements),
    }
    return scene, ground_truth, meta


DIFFICULTY_PRESETS = {
    # Chaque preset RECOPIE CONFIG puis ne modifie que les plages listees ;
    # "hard" ne desactive rien (a la difference de --clean dans l'original
    # qui, lui, mettait tout a zero) : il DURCIT les plages realistes.
    "normal": {},
    "hard": {
        "noise_luma_sigma_range": (5, 12),
        "noise_chroma_sigma_range": (8, 20),
        "dof_blur_coeff_px_per_m": 3.2,
        "focus_error_range_m": (-2.0, 2.0),
        "autofocus_hunt_event_prob": 0.35,
        "vibration_amplitude_px_range": (0.6, 2.2),
        "vignette_strength_range": (0.25, 0.45),
        "jpeg_quality_choices": [55, 60, 65, 70],
        "altitude_min": 3.0, "altitude_max": 7.0,   # marqueurs plus petits
        "cnn_min_visible_fraction": 0.12,
        "sun_elevation_range_deg": (12.0, 75.0),    # ombres plus longues
    },
}


def build_config(difficulty: str) -> dict:
    cfg = dict(CONFIG)
    cfg.update(DIFFICULTY_PRESETS.get(difficulty, {}))
    return cfg


def parse_range(spec: str) -> Tuple[int, int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    n = int(spec)
    return n, n


def main() -> int:
    ap = argparse.ArgumentParser(description="Scenes ArUco realistes (images independantes)")
    ap.add_argument("--n", type=int, default=40, help="nombre de scenes a generer")
    ap.add_argument("--out", default="dataset_realiste")
    ap.add_argument("--markers-per-scene", default="1-3",
                    help="borne 'min-max' du nb de marqueurs par scene")
    ap.add_argument("--marker-ids", default="0-40",
                    help="plage d'IDs 4x4_1000 dans laquelle piocher")
    ap.add_argument("--difficulty", choices=sorted(DIFFICULTY_PRESETS), default="normal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--marker-size-m", type=float, default=None,
                    help="cote physique du marqueur en metres (defaut : CONFIG['marker_real_size'])")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)

    os.makedirs(a.out, exist_ok=True)
    cfg = build_config(a.difficulty)
    if a.marker_size_m is not None:
        cfg["marker_real_size"] = a.marker_size_m
    extra = dict(EXTRA_CONFIG)
    mps_lo, mps_hi = parse_range(a.markers_per_scene)
    id_lo, id_hi = parse_range(a.marker_ids)

    t0 = time.time()
    written = 0
    total_gt = 0
    for i in range(a.n):
        n_markers = random.randint(mps_lo, mps_hi)
        ids = random.sample(range(id_lo, id_hi + 1), min(n_markers, id_hi - id_lo + 1))
        scene, gt, meta = render_independent_scene(ids, cfg, extra)
        if scene is None:
            continue
        name = f"scene_{i:04d}"
        cv2.imwrite(os.path.join(a.out, f"{name}.png"), scene)
        with open(os.path.join(a.out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"markers": gt, "scene": meta}, f, indent=2)
        written += 1
        total_gt += len(gt)
        print(f"\r{written}/{a.n} scenes ({total_gt} marqueurs places)", end="", flush=True)
    dt = time.time() - t0
    print(f"\n{written} scenes ecrites dans {a.out}/ en {dt:.1f}s "
          f"({dt/max(written,1)*1000:.0f} ms/scene), {total_gt} marqueurs au total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
