#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Projet Ardu-Citron : Simulateur de capteur et générateur de dataset ArUco réaliste (Version Multi-processus 4 Coeurs).
Auteur : Spécialiste Vision par Ordinateur & Simulation
"""

import os
import io
import argparse
import cv2
import json
import glob
import math
import random
import re
import numpy as np
from PIL import Image
import cairosvg
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

# Importation du détecteur pour valider le dataset généré
from aruco_detector import ArucoDetector

# ==============================================================================
# CONFIGURATION GLOBALE
# ==============================================================================
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
    # Les deux datasets sont TOTALEMENT SÉPARÉS, avec des générateurs différents :
    #  - "verification_output_dir" : dataset de VÉRIFICATION du programme complet, à base de
    #    trajectoires continues (vol, IMU, dérive, entrée/sortie de cadre). Sert à valider le
    #    système entier dans le temps, PAS à entraîner le CNN.
    #  - "cnn_output_dir" : dataset D'ENTRAÎNEMENT du CNN, exemples 100% indépendants (pas de
    #    trajectoire), répartis directement en train/ et val/ pour la diversité.
    "verification_output_dir": "Dataset_Verification",
    "markers_dir": "Markers_5",

    # --- Dataset CNN (train/val) : exemples indépendants, sans trajectoire (diversité maximale) ---
    "cnn_output_dir": "Dataset_CNN",
    "cnn_examples_per_marker": 600,   # Nb d'exemples indépendants générés par marqueur
    "cnn_val_fraction": 0.15,         # Fraction réservée à la validation (85% train / 15% val)
    "roi_size": (128, 128),           # Taille fixe d'entrée pour le CNN
    "roi_margin_factor": 1.2,         # Marge de base autour du marqueur (contexte visuel)
    "roi_scale_range": (1.0, 1.6),    # Variation de zoom arrière du crop (plans plus ou moins larges)
    "roi_position_jitter_frac": 0.6,  # Décentrage aléatoire du marqueur dans le crop (fraction de la marge)
}

# ==============================================================================
# FONCTIONS DE SIMULATION PHYSIQUE & OPTIQUE
# ==============================================================================

def _multiscale_noise(h, w, scales_sigmas):
    """Bruit à plusieurs échelles spatiales (basse fréquence = mottling, haute fréquence = grain)."""
    total = np.zeros((h, w), dtype=np.float32)
    for scale, sigma in scales_sigmas:
        sh, sw = max(1, h // scale), max(1, w // scale)
        n = np.random.normal(0, sigma, (sh, sw)).astype(np.float32)
        total += cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    return total


def generate_ground_texture(texture_type, size_px, config=None):
    """
    Génère un grand patch de texture de sol (carré size_px x size_px) selon le type demandé.
    Toutes les couleurs sont en BGR (convention OpenCV). Ce patch est ensuite plaqué au sol
    via une véritable homographie 3D (voir compute_ground_homography / warp_ground_texture),
    exactement comme le marqueur, pour que les lattes/brins/dalles/lignes de terrain convergent
    point de fuite en cas de Roll/Pitch.
    """
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
        # Parquet de gymnase : plus clair et plus "glacé" que le bois extérieur (le glacis est
        # géré séparément par apply_specular_highlights, ici on pose juste la teinte de base).
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

        # Rectangle du terrain principal + ligne médiane
        margin_px = int(0.08 * size_px)
        p1 = (margin_px, margin_px)
        p2 = (size_px - margin_px, size_px - margin_px)
        cv2.rectangle(tex, p1, p2, rand_color(), line_w())
        cv2.line(tex, (margin_px, size_px // 2), (size_px - margin_px, size_px // 2), rand_color(), line_w())
        cv2.line(tex, (size_px // 2, margin_px), (size_px // 2, size_px - margin_px), rand_color(), line_w())

        # Cercle central + 2 cercles secondaires (ronds de basket / lancers francs, style générique)
        cv2.circle(tex, (size_px // 2, size_px // 2), int(0.10 * size_px), rand_color(), line_w())
        for cy_c in (margin_px + int(0.18 * size_px), size_px - margin_px - int(0.18 * size_px)):
            cv2.circle(tex, (size_px // 2, cy_c), int(0.07 * size_px), rand_color(), max(2, line_w() - 2))

        # Zones peintes (clés/raquettes) : deux rectangles proches des extrémités
        key_w, key_h = int(0.16 * size_px), int(0.24 * size_px)
        cv2.rectangle(tex, (size_px // 2 - key_w // 2, margin_px),
                      (size_px // 2 + key_w // 2, margin_px + key_h), rand_color(), max(2, line_w() - 1))
        cv2.rectangle(tex, (size_px // 2 - key_w // 2, size_px - margin_px - key_h),
                      (size_px // 2 + key_w // 2, size_px - margin_px), rand_color(), max(2, line_w() - 1))

        # Logo stylisé au centre (forme abstraite, pas de marque réelle)
        if random.random() < cfg.get("gym_logo_prob", 0.6):
            logo_color = rand_color()
            logo_r = int(0.05 * size_px)
            cv2.circle(tex, (size_px // 2, size_px // 2), logo_r, logo_color, -1)
            cv2.circle(tex, (size_px // 2, size_px // 2), int(logo_r * 0.55), tuple(int(c) for c in base), -1)

        # Numéros peints au sol, dispersés
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
        base = np.array([35, 90, 35], dtype=np.float32)  # BGR : vert dominant
        tex = np.tile(base, (h, w, 1))
        blades = _multiscale_noise(h, w, [(2, 18.0), (15, 10.0), (60, 6.0)])[..., None]
        tex += blades * np.array([0.4, 1.0, 0.4], dtype=np.float32)

    elif texture_type == "asphalt":
        base = np.array([55, 55, 58], dtype=np.float32)  # gris sombre neutre
        tex = np.tile(base, (h, w, 1))
        tex += _multiscale_noise(h, w, [(2, 10.0), (30, 4.0)])[..., None]

    elif texture_type == "concrete":
        base = np.array([150, 150, 148], dtype=np.float32)  # gris clair
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
        base = np.array([45, 75, 110], dtype=np.float32)  # brun terre battue
        tex = np.tile(base, (h, w, 1))
        tex += _multiscale_noise(h, w, [(2, 14.0), (20, 10.0), (70, 6.0)])[..., None]

    else:
        tex = np.tile(np.array([80, 80, 80], dtype=np.float32), (h, w, 1))

    tex = np.clip(tex, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(tex, (3, 3), 0)


def compute_ground_homography(R, altitude, drone_x, drone_y, fx, fy, cx, cy):
    """
    Homographie exacte 3x3 mappant un point du sol (X, Y en mètres, dans le même repère
    que les coins du marqueur) vers un pixel de l'image caméra. C'est la même transformation
    géométrique (rotation caméra + perspective) que celle appliquée au marqueur : le sol
    "hérite" donc de la même convergence de point de fuite en cas de Roll/Pitch.
    """
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    A = np.array([
        [1.0, 0.0, -drone_x],
        [0.0, 1.0, -drone_y],
        [0.0, 0.0, altitude],
    ], dtype=np.float64)
    return K @ (R @ A)


def warp_ground_texture(texture, ground_homography, texels_per_meter, width, height):
    """Plaque la texture de sol (indexée en pixels) dans l'image caméra (indexée en pixels)."""
    tex_h, tex_w = texture.shape[:2]
    tex_cx, tex_cy = tex_w / 2.0, tex_h / 2.0
    tpm = float(texels_per_meter)
    # T : pixel de texture -> mètres réels au sol (X, Y)
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


def load_marker(svg_path, size=500):
    try:
        png_bytes = cairosvg.svg2png(url=svg_path, output_width=size, output_height=size)
        pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        return np.array(pil_img)
    except Exception as e:
        print(f"\n[ERREUR] Impossible de charger/convertir {svg_path}: {e}")
        return None


def get_camera_intrinsics(width, height):
    h_fov_rad = np.radians(CONFIG["camera_h_fov"])
    v_fov_rad = np.radians(CONFIG["camera_v_fov"])
    
    fx = width / (2.0 * np.tan(h_fov_rad / 2.0))
    fy = height / (2.0 * np.tan(v_fov_rad / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    
    return fx, fy, cx, cy


def smooth_random_walk(n, dt, tau, sigma, start=0.0):
    walk = np.zeros(n)
    walk[0] = start
    for i in range(1, n):
        drive = random.gauss(0, sigma)
        alpha = dt / tau
        walk[i] = walk[i - 1] + alpha * (drive - walk[i - 1])
    return walk


def low_pass_filter(signal, dt, tau):
    out = np.zeros_like(signal)
    out[0] = signal[0]
    alpha = dt / tau
    for i in range(1, len(signal)):
        out[i] = out[i - 1] + alpha * (signal[i] - out[i - 1])
    return out


def ease_in_out(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))


def generate_flight_trajectory(config, marker_id=None):
    n = config["frames_per_trajectory"]
    duration = config["trajectory_duration_s"]
    dt = duration / n
    t = np.linspace(0.0, duration, n, endpoint=False)
    t_norm = t / duration if duration > 0 else t

    V = config["drone_speed"]
    g = 9.81

    maneuvers = [
        "straight_level",
        "turn_left", "turn_right",
        "climb_straight", "descent_straight",
        "turn_left_climb", "turn_right_climb",
        "turn_left_descent", "turn_right_descent",
        "s_turn", "s_turn_climb", "s_turn_descent", "s_turn_wave",
        "porpoise_wave",
    ]
    maneuver = random.choice(maneuvers)
    is_s_turn = "s_turn" in maneuver
    turning = "turn" in maneuver
    climbing = "climb" in maneuver
    descending = "descent" in maneuver
    waving = "wave" in maneuver

    tau_slow, sigma_slow = config["turbulence_tau_s"], config["turbulence_sigma_deg"]
    tau_fast, sigma_fast = config["turbulence_fast_tau_s"], config["turbulence_fast_sigma_deg"]

    if is_s_turn:
        rate_lo, rate_hi = config["yaw_rate_range_deg_s"]
        target_rate = random.uniform(rate_lo, rate_hi) * random.choice([-1.0, 1.0])
        half_cycles = random.choice(config["s_turn_half_cycles_choices"])
        yaw_rate_profile = target_rate * np.sin(np.pi * half_cycles * t_norm)
    elif turning:
        rate_lo, rate_hi = config["yaw_rate_range_deg_s"]
        target_rate = random.uniform(rate_lo, rate_hi)
        if "right" in maneuver:
            target_rate = -target_rate
        ramp_frac = random.uniform(0.15, 0.35)
        yaw_rate_profile = target_rate * ease_in_out(t_norm / ramp_frac)
    else:
        yaw_rate_profile = np.zeros(n)

    yaw_rate_noise = (smooth_random_walk(n, dt, tau=tau_slow, sigma=sigma_slow * 0.5)
                       + smooth_random_walk(n, dt, tau=tau_fast, sigma=sigma_fast * 0.5))
    yaw_rate = yaw_rate_profile + yaw_rate_noise

    yaw0 = random.uniform(-15.0, 15.0)
    yaw = yaw0 + np.cumsum(yaw_rate) * dt

    yaw_rate_rad = np.radians(yaw_rate_profile)
    target_roll = np.degrees(np.arctan(np.clip(V * yaw_rate_rad / g, -0.9, 0.9)))
    roll = low_pass_filter(target_roll, dt, tau=config["roll_lag_tau_s"])
    roll += smooth_random_walk(n, dt, tau=tau_slow, sigma=sigma_slow)
    roll += smooth_random_walk(n, dt, tau=tau_fast, sigma=sigma_fast)
    roll = np.clip(roll, -config["roll_max_deg"], config["roll_max_deg"])

    alt_min, alt_max = config["altitude_min"], config["altitude_max"]
    alt0 = random.uniform(alt_min, alt_max)

    if waving:
        vspeed_lo, vspeed_hi = config["climb_rate_range_ms"]
        wave_vspeed_amp = random.uniform(vspeed_lo, vspeed_hi)
        cycles_lo, cycles_hi = config["wave_cycles_range"]
        wave_cycles = random.uniform(cycles_lo, cycles_hi)
        climb_rate_profile = wave_vspeed_amp * np.sin(2.0 * np.pi * wave_cycles * t_norm)
        altitude_offset = np.cumsum(climb_rate_profile) * dt
        altitude_offset -= altitude_offset[0]
        altitude = alt0 + altitude_offset
        vspeed_for_pitch = climb_rate_profile
    else:
        if climbing or descending:
            rate_lo, rate_hi = config["climb_rate_range_ms"]
            vspeed = random.uniform(rate_lo, rate_hi) * (1 if climbing else -1)
            alt1 = float(np.clip(alt0 + vspeed * duration, alt_min, alt_max))
        else:
            vspeed = 0.0
            alt1 = alt0
        altitude = alt0 + (alt1 - alt0) * ease_in_out(t_norm)

    altitude += smooth_random_walk(n, dt, tau=tau_slow, sigma=config["altitude_turbulence_sigma_m"])
    altitude += smooth_random_walk(n, dt, tau=tau_fast, sigma=config["altitude_turbulence_sigma_m"] * 0.5)
    altitude = np.clip(altitude, alt_min, alt_max)

    if waving:
        pitch_target = np.degrees(np.arctan2(vspeed_for_pitch, V)) if V > 0 else np.zeros(n)
    else:
        climb_angle_deg = np.degrees(np.arctan2(vspeed, V)) if V > 0 else 0.0
        pitch_target = climb_angle_deg * ease_in_out(t_norm)

    pitch = pitch_target + smooth_random_walk(n, dt, tau=tau_slow, sigma=sigma_slow * 0.6)
    pitch += smooth_random_walk(n, dt, tau=tau_fast, sigma=sigma_fast * 0.6)
    pitch = np.clip(pitch, -config["pitch_max_deg"], config["pitch_max_deg"])

    forward_speed = V + smooth_random_walk(n, dt, tau=0.3, sigma=0.15) \
        + smooth_random_walk(n, dt, tau=tau_fast, sigma=0.08)

    # --- Translation réelle du drone (X, Y) : le marqueur ne reste plus figé à (0,0) ---
    # Un Pitch vers l'avant/arrière change la composante horizontale de la vitesse air ;
    # le Yaw (cap) donne la direction de déplacement dans le plan horizontal.
    yaw_rad = np.radians(yaw)
    horizontal_speed = forward_speed * np.cos(np.radians(pitch))
    vx = horizontal_speed * np.cos(yaw_rad)
    vy = horizontal_speed * np.sin(yaw_rad)

    drift_scale = config.get("position_drift_scale", 1.0)
    pos_x = np.cumsum(vx) * dt * drift_scale
    pos_y = np.cumsum(vy) * dt * drift_scale
    pos_x -= pos_x[0]
    pos_y -= pos_y[0]

    # Dérive latérale due au vent : Random Walk lent, indépendant du cap intentionnel.
    # Peut pousser le marqueur dans les coins de l'image, voire hors du cadre.
    gust_x = smooth_random_walk(n, dt, tau=config["wind_gust_tau_s"], sigma=config["wind_gust_sigma_m"])
    gust_y = smooth_random_walk(n, dt, tau=config["wind_gust_tau_s"], sigma=config["wind_gust_sigma_m"])
    pos_x = pos_x + gust_x
    pos_y = pos_y + gust_y

    # --- Point d'entrée : le marqueur ne démarre plus centré, il apparaît d'un côté de l'image ---
    # Calculé à partir du rayon d'empreinte au sol (altitude x FOV) et du cap moyen de la trajectoire,
    # avec une dispersion angulaire pour varier le côté/l'angle d'entrée d'une trajectoire à l'autre.
    width, height = config["output_resolution"]
    fx0, fy0, _, _ = get_camera_intrinsics(width, height)
    half_diag_px = np.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2)
    footprint_radius_m = (half_diag_px / fx0) * alt0

    total_dx = pos_x[-1] - pos_x[0]
    total_dy = pos_y[-1] - pos_y[0]
    if abs(total_dx) + abs(total_dy) > 1e-6:
        heading_angle = np.arctan2(total_dy, total_dx)
    else:
        heading_angle = np.radians(yaw0)

    angle_jitter = np.radians(random.uniform(-config["entry_angle_jitter_deg"], config["entry_angle_jitter_deg"]))
    entry_angle = heading_angle + angle_jitter
    entry_radius = footprint_radius_m * random.uniform(*config["entry_touch_factor_range"])

    entry_x = entry_radius * np.cos(entry_angle)
    entry_y = entry_radius * np.sin(entry_angle)
    pos_x = pos_x - entry_x
    pos_y = pos_y - entry_y

    return {
        "t": t,
        "altitude": altitude,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "yaw_rate": yaw_rate,
        "forward_speed": forward_speed,
        "pos_x": pos_x,
        "pos_y": pos_y,
        "maneuver": maneuver,
    }


def simulate_mpu6050_imu(traj, dt, config):
    n = len(traj["roll"])
    roll_true = traj["roll"]
    pitch_true = traj["pitch"]
    yaw_true = traj["yaw"]

    roll_rate = np.gradient(roll_true, dt)
    pitch_rate = np.gradient(pitch_true, dt)
    yaw_rate = np.gradient(yaw_true, dt)

    roll_r = np.radians(roll_true)
    pitch_r = np.radians(pitch_true)
    roll_rate_r = np.radians(roll_rate)
    pitch_rate_r = np.radians(pitch_rate)
    yaw_rate_r = np.radians(yaw_rate)

    p = roll_rate_r - yaw_rate_r * np.sin(pitch_r)
    q = pitch_rate_r * np.cos(roll_r) + yaw_rate_r * np.cos(pitch_r) * np.sin(roll_r)
    r = -pitch_rate_r * np.sin(roll_r) + yaw_rate_r * np.cos(pitch_r) * np.cos(roll_r)

    p_deg, q_deg, r_deg = np.degrees(p), np.degrees(q), np.degrees(r)

    bias_lo, bias_hi = config["imu_gyro_bias_init_range_dps"]
    bias_x = smooth_random_walk(n, dt, tau=config["imu_gyro_bias_walk_tau_s"],
                                 sigma=config["imu_gyro_bias_walk_sigma_dps"],
                                 start=random.uniform(bias_lo, bias_hi))
    bias_y = smooth_random_walk(n, dt, tau=config["imu_gyro_bias_walk_tau_s"],
                                 sigma=config["imu_gyro_bias_walk_sigma_dps"],
                                 start=random.uniform(bias_lo, bias_hi))
    bias_z = smooth_random_walk(n, dt, tau=config["imu_gyro_bias_walk_tau_s"],
                                 sigma=config["imu_gyro_bias_walk_sigma_dps"],
                                 start=random.uniform(bias_lo, bias_hi))

    gyro_noise = config["imu_gyro_noise_density_dps"]
    gyro_x = p_deg + bias_x + np.random.normal(0, gyro_noise, n)
    gyro_y = q_deg + bias_y + np.random.normal(0, gyro_noise, n)
    gyro_z = r_deg + bias_z + np.random.normal(0, gyro_noise, n)

    accel_noise_deg = config["imu_accel_noise_sigma_dps"]
    attenuation = config["imu_accel_roll_attenuation"]
    roll_accel_true = roll_true * attenuation
    pitch_accel_true = pitch_true

    roll_accel = roll_accel_true + np.random.normal(0, accel_noise_deg, n)
    pitch_accel = pitch_accel_true + np.random.normal(0, accel_noise_deg, n)

    roll_accel_r = np.radians(roll_accel)
    pitch_accel_r = np.radians(pitch_accel)
    accel_x = -np.sin(pitch_accel_r)
    accel_y = np.sin(roll_accel_r) * np.cos(pitch_accel_r)
    accel_z = np.cos(roll_accel_r) * np.cos(pitch_accel_r)

    alpha = config["imu_complementary_alpha"]
    err_lo, err_hi = config["imu_initial_attitude_error_deg"]

    def initial_error():
        return random.choice([-1.0, 1.0]) * random.uniform(err_lo, err_hi)

    roll_imu = np.zeros(n)
    pitch_imu = np.zeros(n)
    yaw_imu = np.zeros(n)

    roll_imu[0] = roll_true[0] + initial_error()
    pitch_imu[0] = pitch_true[0] + initial_error()
    yaw_imu[0] = yaw_true[0] + initial_error()

    for i in range(1, n):
        roll_pred = roll_imu[i - 1] + gyro_x[i] * dt
        pitch_pred = pitch_imu[i - 1] + gyro_y[i] * dt
        yaw_pred = yaw_imu[i - 1] + gyro_z[i] * dt

        roll_imu[i] = alpha * roll_pred + (1.0 - alpha) * roll_accel[i]
        pitch_imu[i] = alpha * pitch_pred + (1.0 - alpha) * pitch_accel[i]
        yaw_imu[i] = yaw_pred

    return {
        "roll_imu": roll_imu, "pitch_imu": pitch_imu, "yaw_imu": yaw_imu,
        "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z,
        "accel_x": accel_x, "accel_y": accel_y, "accel_z": accel_z,
    }


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
    """
    Projette le marqueur (et son ombre) dans l'image caméra en tenant compte de :
    - l'attitude du drone (roll/pitch/yaw),
    - sa position réelle (drone_x, drone_y) par rapport au marqueur (dérive de vol + vent),
    - la direction du soleil (sun_az_deg, sun_elev_deg) pour l'ombre portée.
    Renvoie également l'homographie 3x3 du plan sol, pour que le sol entier (textures,
    lattes, dalles...) subisse exactement la même transformation géométrique.
    """
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

    # Ombre portée : longueur et direction dépendent de l'altitude et du vecteur soleil.
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

    # Pénombre : l'ombre devient plus floue à mesure que l'altitude augmente.
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
    """
    Distorsion optique complète (modèle Brown-Conrady) : radiale (k1, k2, k3) + tangentielle
    (p1, p2, due au léger désalignement lentille/capteur). k1 seul ne modélisait qu'un
    barrel/pincushion simple ; k2/k3 affinent la courbe aux bords, p1/p2 introduisent
    une asymétrie (l'image n'est plus symétrique en distorsion).
    """
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
    """Défauts capteur fixes (hot pixels) : mêmes positions sur toute une trajectoire, comme un vrai capteur."""
    mask = np.zeros((height, width), dtype=bool)
    n = random.randint(*count_range)
    for _ in range(n):
        yx = (random.randint(0, height - 1), random.randint(0, width - 1))
        mask[yx] = True
    return mask


def apply_sensor_noise(image, config, hot_pixel_mask=None):
    """
    Bruit capteur IMX708 réaliste :
    - bruit de photon (Poisson/shot noise) proportionnel à sqrt(intensité),
    - bruit de chrominance plus fort que la luminance, amplifié en basse lumière,
    - pixels chauds fixes (défauts capteur).
    """
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
    """Assombrissement progressif et radial vers les coins (typique des optiques légères)."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    ccx, ccy = w / 2.0, h / 2.0
    max_r = np.sqrt(ccx ** 2 + ccy ** 2)
    r = np.sqrt((xx - ccx) ** 2 + (yy - ccy) ** 2) / max_r
    mask = np.clip(1.0 - strength * (r ** 2), 0.0, 1.0).astype(np.float32)
    out = image.astype(np.float32) * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_auto_exposure(image, gain):
    """Simule les sautes de gain/exposition automatique de la caméra, frame à frame."""
    out = image.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_focus_blur(image, ksize):
    """Flou gaussien dont l'intensité (ksize) code la profondeur de champ / le défaut de mise au point."""
    ksize = int(round(ksize))
    if ksize < 3:
        return image
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def apply_specular_highlights(image, config):
    """
    Reflets spéculaires du parquet ciré : quelques traînées lumineuses (néons/soleil réfléchis)
    qui peuvent saturer localement une partie de l'image, y compris sur le marqueur lui-même.
    """
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
    """
    Éclairage de gymnase (néons/LED sur secteur redressé -> scintillement ~100Hz + légère
    dominante verte typique des tubes fluorescents) combiné à la balance des blancs de la
    caméra (dérive lente froid/chaud/verdâtre pendant la trajectoire).
    """
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
    """Micro-tremblement mécanique (moteur/hélice/servos) : léger décalage image par image."""
    if amplitude_px <= 0:
        return image
    dx = np.random.normal(0, amplitude_px)
    dy = np.random.normal(0, amplitude_px)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def apply_jpeg_compression(image, quality_choices):
    """Simule la compression JPEG embarquée de la caméra (artefacts de blocs, perte chroma)."""
    quality = random.choice(quality_choices)
    ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_marker_imperfections(marker_rgba, config):
    """
    Marqueur physique imparfait : contraste d'impression imparfait (noir/blanc non purs),
    grain papier, léger gondolement (papier non plan), coin décollé/froissé. Calculé UNE FOIS
    par trajectoire (le même marqueur physique reste le même tout au long du vol).
    """
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


def render_frame(marker_base, ground_texture, width, height, fx,
                  altitude, roll, pitch, yaw, yaw_rate, speed, drone_x, drone_y,
                  sun_az_deg, sun_elev_deg, focus_distance_m, autofocus_hunting,
                  k1, k2, k3, p1, p2, wb_temp, wb_green, t_abs, hot_pixel_mask, config):
    """
    Rend une frame complète (sol + marqueur + tous les effets caméra/optiques/environnementaux).
    Fonction PARTAGÉE entre le dataset de VÉRIFICATION (trajectoires continues, voir
    process_verification_marker) et le dataset CNN (exemples indépendants, voir
    generate_cnn_examples_for_marker) : la physique du rendu est strictement identique dans
    les deux cas, seule la façon dont les paramètres d'entrée sont générés diffère
    (progression temporelle d'un vol vs tirage aléatoire indépendant par exemple).
    """
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


def extract_marker_id(filename):
    match = re.search(r'4x4_1000-(\d+)\.svg', os.path.basename(filename))
    return int(match.group(1)) if match else 0


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='█'):
    percent = ("{0:." + str(decimals) + f"f}}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

# ==============================================================================
# WORKER PROCESS: TRAITEMENT D'UN FICHIER SVG (1 TRAJECTOIRE COMPLÈTE)
# ==============================================================================

def process_verification_marker(svg_path):
    """
    Dataset de VÉRIFICATION du programme complet : génère plusieurs trajectoires (vols)
    continues pour un même marqueur (dynamique de vol, IMU, dérive, entrée/sortie de cadre).
    Sert à valider le système entier (tracking, fusion IMU, comportement dans le temps) —
    PAS à entraîner le CNN (voir generate_cnn_examples_for_marker pour ça).
    """
    # Sécurise l'aléa pour éviter les doublons de trajectoire entre processus
    random.seed(os.getpid() + int(time.time() * 1000) % 1000)
    np.random.seed(os.getpid() + int(time.time() * 1000) % 1000)

    marker_id = extract_marker_id(svg_path)
    marker_base_clean = load_marker(svg_path, size=600)

    if marker_base_clean is None:
        return 0, 0

    output_dir = CONFIG["verification_output_dir"]
    frames_per_traj = CONFIG["frames_per_trajectory"]
    width, height = CONFIG["output_resolution"]
    fx, _, _, _ = get_camera_intrinsics(width, height)
    dt_frame = CONFIG["trajectory_duration_s"] / frames_per_traj

    local_detector = ArucoDetector()
    local_detected_count = 0
    total_written = 0

    n_trajectories = CONFIG["trajectories_per_marker"]

    for traj_idx in range(n_trajectories):
        traj = generate_flight_trajectory(CONFIG, marker_id=marker_id)
        trajectory_id = f"traj_{marker_id}_{traj_idx}_{int(random.random() * 1e6):06d}"
        imu = simulate_mpu6050_imu(traj, dt_frame, CONFIG)

        # --- Environnement de la trajectoire (fixe pendant tout le vol) ---
        texture_type = random.choices(
            CONFIG["ground_texture_types"], weights=CONFIG.get("ground_texture_weights"), k=1
        )[0]
        tex_size_px = int(CONFIG["ground_texture_size_m"] * CONFIG["ground_texels_per_meter"])
        ground_texture = generate_ground_texture(texture_type, tex_size_px, CONFIG)

        sun_az_deg = random.uniform(0.0, 360.0)
        sun_elev_deg = random.uniform(*CONFIG["sun_elevation_range_deg"])

        alt0 = float(traj["altitude"][0])
        focus_distance_m = float(np.clip(
            alt0 + random.uniform(*CONFIG["focus_error_range_m"]),
            CONFIG["altitude_min"], CONFIG["altitude_max"]
        ))

        hunt_start, hunt_end = -1, -1
        if random.random() < CONFIG["autofocus_hunt_event_prob"]:
            hunt_len = random.randint(*CONFIG["autofocus_hunt_len_frames_range"])
            hunt_start = random.randint(0, max(0, frames_per_traj - 1))
            hunt_end = min(frames_per_traj, hunt_start + hunt_len)

        # Marqueur physique imparfait (papier réel) : identique sur toute la trajectoire
        marker_base = apply_marker_imperfections(marker_base_clean, CONFIG)

        # Pixels chauds fixes (défaut capteur) : identiques sur toute la trajectoire
        hot_pixel_mask = generate_hot_pixel_mask(width, height, CONFIG["hot_pixel_count_range"])

        # Distorsion optique complète tirée une fois (la lentille ne change pas en vol)
        k1 = random.uniform(*CONFIG["k1_distortion_range"])
        k2 = random.uniform(*CONFIG["k2_distortion_range"])
        k3 = random.uniform(*CONFIG["k3_distortion_range"])
        p1 = random.uniform(*CONFIG["p1_distortion_range"])
        p2 = random.uniform(*CONFIG["p2_distortion_range"])

        # Balance des blancs : dérive lente (froid/chaud/verdâtre) le long de la trajectoire
        wb_temp0 = random.uniform(*CONFIG["wb_temp_range"])
        wb_green0 = random.uniform(*CONFIG["wb_green_range"])
        wb_temp_path = wb_temp0 + smooth_random_walk(
            frames_per_traj, dt_frame, tau=CONFIG["wb_drift_tau_s"], sigma=CONFIG["wb_drift_sigma"]
        )
        wb_green_path = wb_green0 + smooth_random_walk(
            frames_per_traj, dt_frame, tau=CONFIG["wb_drift_tau_s"], sigma=CONFIG["wb_drift_sigma"] * 0.5
        )

        # Instant de référence pour le scintillement néon (déphasage aléatoire par trajectoire)
        t0_abs = random.uniform(0.0, 10.0)

        has_appeared = False

        for img_idx in range(frames_per_traj):
            altitude = float(traj["altitude"][img_idx])
            roll = float(traj["roll"][img_idx])
            pitch = float(traj["pitch"][img_idx])
            yaw = float(traj["yaw"][img_idx])
            yaw_rate = float(traj["yaw_rate"][img_idx])
            speed = float(traj["forward_speed"][img_idx])
            drone_x = float(traj["pos_x"][img_idx])
            drone_y = float(traj["pos_y"][img_idx])

            # Test de présence dans le cadre AVANT le rendu complet (pour l'arrêt anticipé)
            R_check = compute_rotation_matrix(roll, pitch, yaw)
            fx_c, fy_c, cx_c, cy_c = get_camera_intrinsics(width, height)
            s_chk = CONFIG["marker_real_size"]
            base_xy_chk = np.array([[-s_chk/2, -s_chk/2], [s_chk/2, -s_chk/2], [s_chk/2, s_chk/2], [-s_chk/2, s_chk/2]])
            pts_w_chk = np.column_stack([base_xy_chk[:, 0] - drone_x, base_xy_chk[:, 1] - drone_y, np.full(4, altitude)])
            pts_chk = project_points(pts_w_chk, R_check, fx_c, fy_c, cx_c, cy_c)
            xs, ys = pts_chk[:, 0], pts_chk[:, 1]
            overlaps_frame = (xs.max() >= 0) and (xs.min() <= width - 1) and (ys.max() >= 0) and (ys.min() <= height - 1)
            marker_fully_in_frame = bool(
                (xs.min() >= 0) and (xs.max() <= width - 1) and (ys.min() >= 0) and (ys.max() <= height - 1)
            )

            if overlaps_frame:
                has_appeared = True
            elif has_appeared:
                # Le marqueur est sorti du cadre après y être apparu : on arrête cette trajectoire ici.
                break

            autofocus_hunting = hunt_start <= img_idx < hunt_end
            t_abs = t0_abs + img_idx * dt_frame

            scene, pts_img, ground_H, meta = render_frame(
                marker_base, ground_texture, width, height, fx,
                altitude, roll, pitch, yaw, yaw_rate, speed, drone_x, drone_y,
                sun_az_deg, sun_elev_deg, focus_distance_m, autofocus_hunting,
                k1, k2, k3, p1, p2, float(wb_temp_path[img_idx]), float(wb_green_path[img_idx]),
                t_abs, hot_pixel_mask, CONFIG
            )

            detected_markers = local_detector.detect(scene)
            is_detected = any(m["id"] == int(marker_id) for m in detected_markers)
            if is_detected:
                local_detected_count += 1

            marker_size_px = calculate_marker_size(pts_img)

            filename_base = f"marker_{marker_id}_t{traj_idx}_{img_idx:03d}"
            png_path = os.path.join(output_dir, f"{filename_base}.png")
            cv2.imwrite(png_path, scene)

            json_data = {
                "marker_id": int(marker_id),
                "trajectory_id": trajectory_id,
                "maneuver": traj["maneuver"],
                "frame_index": int(img_idx),
                "timestamp_s": round(float(img_idx * dt_frame), 4),
                "distance_m": round(float(altitude), 2),
                "roll_deg": round(float(roll), 1),
                "pitch_deg": round(float(pitch), 1),
                "yaw_deg": round(float(yaw), 1),
                "yaw_rate_deg_s": round(float(yaw_rate), 2),
                "marker_pixels": int(marker_size_px),
                "marker_fully_in_frame": marker_fully_in_frame,
                "aruco_corners": pts_img.tolist(),
                "drone_speed_ms": round(float(speed), 2),
                "drone_pos_m": [round(drone_x, 3), round(drone_y, 3)],
                "camera": "Raspberry Pi Camera v3",
                "detected_by_bench_detector": bool(is_detected),

                "environment": {
                    "ground_texture": texture_type,
                    "sun_azimuth_deg": round(sun_az_deg, 1),
                    "sun_elevation_deg": round(sun_elev_deg, 1),
                    "focus_distance_m": round(focus_distance_m, 2),
                    "autofocus_hunting": bool(autofocus_hunting),
                    "ae_gain": round(float(meta["ae_gain"]), 3),
                    "vignette_strength": round(float(meta["vignette_strength"]), 3),
                    "vibration_px": round(float(meta["vibration_px"]), 3),
                    "wb_temp": round(float(wb_temp_path[img_idx]), 3),
                    "wb_green": round(float(wb_green_path[img_idx]), 3),
                    "lens_distortion": {"k1": round(k1, 4), "k2": round(k2, 4), "k3": round(k3, 4),
                                        "p1": round(p1, 5), "p2": round(p2, 5)},
                },

                "imu_mpu6050": {
                    "roll_deg": round(float(imu["roll_imu"][img_idx]), 2),
                    "pitch_deg": round(float(imu["pitch_imu"][img_idx]), 2),
                    "yaw_deg": round(float(imu["yaw_imu"][img_idx]), 2),
                    "gyro_dps": [
                        round(float(imu["gyro_x"][img_idx]), 4),
                        round(float(imu["gyro_y"][img_idx]), 4),
                        round(float(imu["gyro_z"][img_idx]), 4)
                    ],
                    "accel_g": [
                        round(float(imu["accel_x"][img_idx]), 4),
                        round(float(imu["accel_y"][img_idx]), 4),
                        round(float(imu["accel_z"][img_idx]), 4)
                    ]
                }
            }

            json_path = os.path.join(output_dir, f"{filename_base}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)

            total_written += 1

    return total_written, local_detected_count

# ==============================================================================
# ÉTAPE 2 : DATASET CNN (train/val) — EXEMPLES INDÉPENDANTS, SANS TRAJECTOIRE
# ==============================================================================
# Contrairement au dataset de VÉRIFICATION ci-dessus (trajectoires continues, IMU, dérive
# temporelle), ce dataset ne modélise AUCUN vol : chaque exemple est un tirage indépendant
# (altitude, attitude, position du marqueur dans l'image, environnement, éclairage, optique,
# imperfections du marqueur) rendu directement en crop ROI. Objectif : diversité maximale
# pour l'entraînement du CNN, pas cohérence temporelle. Écrit directement en train/ et val/.

def generate_cnn_examples_for_marker(svg_path):
    """Génère cnn_examples_per_marker exemples indépendants pour un marqueur, répartis train/val."""
    random.seed(os.getpid() + int(time.time() * 1000) % 1000 + 7919)
    np.random.seed(os.getpid() + int(time.time() * 1000) % 1000 + 7919)

    marker_id = extract_marker_id(svg_path)
    marker_base_clean = load_marker(svg_path, size=600)
    if marker_base_clean is None:
        return 0, 0

    width, height = CONFIG["output_resolution"]
    fx, fy, cx, cy = get_camera_intrinsics(width, height)
    half_diag_px = np.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2)

    roi_size = CONFIG["roi_size"]
    margin_factor = CONFIG["roi_margin_factor"]
    scale_lo, scale_hi = CONFIG["roi_scale_range"]
    position_jitter_frac = CONFIG["roi_position_jitter_frac"]
    val_fraction = CONFIG["cnn_val_fraction"]
    n_examples = CONFIG["cnn_examples_per_marker"]

    train_dir = os.path.join(CONFIG["cnn_output_dir"], "train")
    val_dir = os.path.join(CONFIG["cnn_output_dir"], "val")

    n_train, n_val = 0, 0

    for ex_idx in range(n_examples):
        # --- Tirage 100% indépendant : pas de dynamique de vol, juste de la diversité ---
        altitude = random.uniform(CONFIG["altitude_min"], CONFIG["altitude_max"])
        roll = random.uniform(-CONFIG["roll_max_deg"], CONFIG["roll_max_deg"])
        pitch = random.uniform(-CONFIG["pitch_max_deg"], CONFIG["pitch_max_deg"])
        yaw = random.uniform(0.0, 360.0)
        yaw_rate = random.gauss(0.0, 8.0)
        speed = max(0.0, CONFIG["drone_speed"] + random.gauss(0.0, 1.0))

        # Position du marqueur dans l'image : du centre jusqu'à (voire au-delà de) l'empreinte
        # au sol -> diversité de cadrage (centré, coin, partiellement hors-champ).
        footprint_radius_m = (half_diag_px / fx) * altitude
        place_radius = footprint_radius_m * random.uniform(0.0, 1.15)
        place_angle = random.uniform(0.0, 2 * np.pi)
        drone_x = -place_radius * np.cos(place_angle)
        drone_y = -place_radius * np.sin(place_angle)

        texture_type = random.choices(
            CONFIG["ground_texture_types"], weights=CONFIG.get("ground_texture_weights"), k=1
        )[0]
        tex_size_px = int(CONFIG["ground_texture_size_m"] * CONFIG["ground_texels_per_meter"])
        ground_texture = generate_ground_texture(texture_type, tex_size_px, CONFIG)

        sun_az_deg = random.uniform(0.0, 360.0)
        sun_elev_deg = random.uniform(*CONFIG["sun_elevation_range_deg"])
        focus_distance_m = float(np.clip(
            altitude + random.uniform(*CONFIG["focus_error_range_m"]),
            CONFIG["altitude_min"], CONFIG["altitude_max"]
        ))
        autofocus_hunting = random.random() < CONFIG["autofocus_hunt_event_prob"]

        marker_base = apply_marker_imperfections(marker_base_clean, CONFIG)
        hot_pixel_mask = generate_hot_pixel_mask(width, height, CONFIG["hot_pixel_count_range"])

        k1 = random.uniform(*CONFIG["k1_distortion_range"])
        k2 = random.uniform(*CONFIG["k2_distortion_range"])
        k3 = random.uniform(*CONFIG["k3_distortion_range"])
        p1 = random.uniform(*CONFIG["p1_distortion_range"])
        p2 = random.uniform(*CONFIG["p2_distortion_range"])
        wb_temp = random.uniform(*CONFIG["wb_temp_range"])
        wb_green = random.uniform(*CONFIG["wb_green_range"])
        t_abs = random.uniform(0.0, 10.0)

        scene, pts_img, ground_H, meta = render_frame(
            marker_base, ground_texture, width, height, fx,
            altitude, roll, pitch, yaw, yaw_rate, speed, drone_x, drone_y,
            sun_az_deg, sun_elev_deg, focus_distance_m, autofocus_hunting,
            k1, k2, k3, p1, p2, wb_temp, wb_green, t_abs, hot_pixel_mask, CONFIG
        )

        # --- Crop ROI directement à la génération (pas de fichier caméra intermédiaire) ---
        xs, ys = pts_img[:, 0], pts_img[:, 1]
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
        box_w, box_h = max(xmax - xmin, 1), max(ymax - ymin, 1)
        margin = int(max(box_w, box_h) * margin_factor)

        scale_modifier = random.uniform(scale_lo, scale_hi)
        current_margin = int(margin * scale_modifier)
        max_jitter = int(current_margin * position_jitter_frac)
        dx = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0
        dy = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0

        crop_xmin = max(0, center_x - current_margin + dx)
        crop_xmax = min(width, center_x + current_margin + dx)
        crop_ymin = max(0, center_y - current_margin + dy)
        crop_ymax = min(height, center_y + current_margin + dy)

        roi = scene[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        if roi.size == 0:
            continue
        roi_resized = cv2.resize(roi, roi_size)

        is_val = random.random() < val_fraction
        split_dir = val_dir if is_val else train_dir
        base_name = f"marker_{marker_id}_{ex_idx:05d}"

        cv2.imwrite(os.path.join(split_dir, f"{base_name}.png"), roi_resized)

        output_data = {
            "marker_id": int(marker_id),
            "split": "val" if is_val else "train",
            "img_channels_height_width": [3, roi_size[0], roi_size[1]],
            "target_pose_xyz_rpy": [
                round(float(drone_x), 4), round(float(drone_y), 4), round(float(altitude), 4),
                round(float(roll), 2), round(float(pitch), 2), round(float(yaw), 2)
            ],
        }
        with open(os.path.join(split_dir, f"{base_name}.json"), 'w', encoding='utf-8') as out_f:
            json.dump(output_data, out_f, indent=4)

        if is_val:
            n_val += 1
        else:
            n_train += 1

    return n_train, n_val


def main():
    parser = argparse.ArgumentParser(description="Ardu-Citron Dataset Generator (Caméra + ROI CNN)")
    parser.add_argument("--skip-verification", action="store_true",
                         help="Ne pas générer le dataset de vérification (trajectoires)")
    parser.add_argument("--skip-cnn", action="store_true",
                         help="Ne pas générer le dataset CNN (train/val, exemples indépendants)")
    args = parser.parse_args()

    print("=== [Ardu-Citron Dataset Generator - Multi-processus 4 Coeurs] ===")
    print("    2 datasets TOTALEMENT SÉPARÉS : vérification (trajectoires) et CNN (train/val, sans trajectoire)\n")

    svg_files = glob.glob("4x4_1000-*.svg")
    if not svg_files:
        svg_files = glob.glob(os.path.join(CONFIG["markers_dir"], "4x4_1000-*.svg"))
    if not svg_files:
        print(f"\n[ERREUR] Aucun fichier '4x4_1000-*.svg' trouvé.")
        return
    svg_files = sorted(svg_files, key=extract_marker_id)
    num_markers = len(svg_files)
    print(f"-> {num_markers} marqueurs détectés.")

    # ==========================================================================
    # [1/2] DATASET DE VÉRIFICATION : trajectoires continues (vol, IMU, dérive...)
    # ==========================================================================
    if not args.skip_verification:
        verif_dir = CONFIG["verification_output_dir"]
        os.makedirs(verif_dir, exist_ok=True)

        frames_per_traj = CONFIG["frames_per_trajectory"]
        n_traj = CONFIG["trajectories_per_marker"]
        total_images_max = num_markers * n_traj * frames_per_traj

        print(f"-> [1/2] VÉRIFICATION ('{verif_dir}/') : {num_markers} marqueurs x {n_traj} trajectoires x "
              f"jusqu'à {frames_per_traj} frames (le marqueur peut sortir du cadre avant la fin -> "
              f"trajectoire arrêtée plus tôt), soit au maximum {total_images_max} images.")

        start_time = time.time()
        total_processed = 0
        total_detected = 0

        print_progress_bar(0, total_images_max, prefix='Progression:', suffix='Terminé', length=50)

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_verification_marker, svg_path): svg_path for svg_path in svg_files}
            for future in as_completed(futures):
                try:
                    processed, detected = future.result()
                    total_processed += processed
                    total_detected += detected
                    print_progress_bar(min(total_processed, total_images_max), total_images_max,
                                        prefix='Progression:', suffix='Terminé', length=50)
                except Exception as e:
                    svg_path = futures[future]
                    print(f"\n[ERREUR] Le processus pour {svg_path} a planté : {e}")
        print()

        elapsed_time = time.time() - start_time
        success_rate = (total_detected / total_processed) * 100 if total_processed > 0 else 0

        print(f"[SUCCÈS] Dataset de VÉRIFICATION généré dans '{verif_dir}/' !")
        print(f"📸 {total_processed} images réellement écrites (sur {total_images_max} au maximum théorique).")
        print(f"⏱️ Temps : {elapsed_time:.2f}s ({total_processed / elapsed_time:.1f} images/s)" if elapsed_time > 0 else "")
        print(f"🎯 Détectabilité ArucoDetector : {success_rate:.2f} % ({total_detected}/{total_processed})\n")
    else:
        print("-> [1/2] Dataset de vérification ignoré (--skip-verification).")

    # ==========================================================================
    # [2/2] DATASET CNN : exemples 100% indépendants, sans trajectoire, train/val direct
    # ==========================================================================
    if not args.skip_cnn:
        cnn_dir = CONFIG["cnn_output_dir"]
        train_dir = os.path.join(cnn_dir, "train")
        val_dir = os.path.join(cnn_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        n_examples = CONFIG["cnn_examples_per_marker"]
        total_cnn = num_markers * n_examples
        print(f"-> [2/2] CNN ('{cnn_dir}/train' + '{cnn_dir}/val') : {num_markers} marqueurs x {n_examples} "
              f"exemples indépendants (crops {CONFIG['roi_size'][0]}x{CONFIG['roi_size'][1]}) = {total_cnn} images, "
              f"~{CONFIG['cnn_val_fraction']*100:.0f}% en validation.")

        start_time = time.time()
        total_train, total_val = 0, 0

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(generate_cnn_examples_for_marker, svg_path): svg_path for svg_path in svg_files}
            for future in as_completed(futures):
                try:
                    n_train, n_val = future.result()
                    total_train += n_train
                    total_val += n_val
                except Exception as e:
                    svg_path = futures[future]
                    print(f"\n[ERREUR] Le processus CNN pour {svg_path} a planté : {e}")

        elapsed_time = time.time() - start_time
        total_cnn_written = total_train + total_val

        print(f"[SUCCÈS] Dataset CNN généré : {total_train} train / {total_val} val ({total_cnn_written} images).")
        print(f"⏱️ Temps : {elapsed_time:.2f}s ({total_cnn_written / elapsed_time:.1f} images/s)" if elapsed_time > 0 else "")
    else:
        print("-> [2/2] Dataset CNN ignoré (--skip-cnn).")

    print(f"\n📦 Dataset VÉRIFICATION (trajectoires) : '{CONFIG['verification_output_dir']}/'")
    print(f"📦 Dataset CNN (train/val, indépendant) : '{CONFIG['cnn_output_dir']}/train' + '/val'")


if __name__ == "__main__":
    # Indispensable pour la portabilité cross-platform des sous-processus
    multiprocessing.freeze_support()
    main()
