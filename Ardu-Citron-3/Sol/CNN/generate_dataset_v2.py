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
    "k1_distortion": -0.07,        # Coefficient radial k1 de la lentille (IMX708)
    
    # --- Géométrie de vol : translation réelle du drone (dérive du marqueur dans l'image) ---
    "position_drift_scale": 1.0,     # Facteur d'échelle sur la vitesse horizontale intégrée (X,Y du drone)
    "wind_gust_sigma_m": 0.35,       # Amplitude de la dérive latérale due au vent (Random Walk, en mètres)
    "wind_gust_tau_s": 0.40,         # Constante de temps de la dérive du vent (corrélation lente)

    # --- Rendu du sol : textures hétérogènes + homographie complète ---
    "ground_texture_types": ["wood", "grass", "asphalt", "concrete", "tile", "dirt"],
    "ground_texture_size_m": 24.0,   # Taille physique du patch de texture généré (mètres, carré)
    "ground_texels_per_meter": 50,   # Résolution de la texture (pixels de texture par mètre réel)

    # --- Éclairage : position du soleil, ombre réaliste, AE, vignetage ---
    "sun_elevation_range_deg": (25.0, 75.0),   # Hauteur du soleil dans le ciel (degrés)
    "shadow_length_coeff": 0.12,     # Coefficient reliant altitude drone -> longueur d'ombre projetée
    "shadow_blur_base_px": 5,        # Flou de base de l'ombre (pixels)
    "shadow_blur_altitude_coeff": 3.0,  # L'ombre devient plus floue (pénombre) quand l'altitude augmente
    "ae_gain_range": (0.7, 1.3),      # Gain d'auto-exposition appliqué frame à frame
    "vignette_strength_range": (0.15, 0.35),  # Intensité du vignetage optique

    # --- Bruit capteur (chrominance) & profondeur de champ ---
    "noise_luma_sigma_range": (2, 8),     # Bruit gaussien sur la luminance (Y)
    "noise_chroma_sigma_range": (4, 14),  # Bruit plus fort sur la chrominance (Cr/Cb), typique petits capteurs
    "chroma_lowlight_boost": 1.5,         # Amplification du bruit chroma dans les zones sombres
    "focus_error_range_m": (-1.0, 1.0),   # Erreur de mise au point autofocus par rapport à l'altitude initiale
    "dof_blur_base_px": 3,                # Flou de base (mise au point parfaite)
    "dof_blur_coeff_px_per_m": 2.0,       # Flou additionnel par mètre d'écart à la distance de mise au point
    "dof_max_ksize": 11,                  # Taille max du noyau de flou (px)
    "autofocus_hunt_event_prob": 0.15,    # Probabilité d'un "saut" de mise au point pendant la trajectoire
    "autofocus_hunt_len_frames_range": (15, 60),  # Durée (en frames) d'un saut de mise au point
    "autofocus_hunt_extra_ksize": 6,      # Flou additionnel pendant un saut de mise au point

    # Intensité des effets
    "autofocus_blur_prob": 0.3,    # (conservé pour compatibilité, non utilisé par le nouveau pipeline DOF)

    # Chemins des fichiers
    "output_dir": "Dataset",
    "markers_dir": "Markers_5",

    # --- Dataset ROI pour CNN (fusion de Prepare.py) ---
    "roi_output_dir": "cnn_roi_dataset",
    "roi_size": (128, 128),           # Taille fixe d'entrée pour le CNN
    "roi_augmentation_factor": 5,     # 7500 images caméra x 5
    "roi_margin_factor": 1.2,         # Marge autour du marqueur (contexte visuel)
    "roi_jitter_px": 20,              # Bruit de tracking simulé (translation aléatoire du crop)
    "roi_scale_range": (1.0, 1.4),    # Variation de zoom arrière du crop (plans plus larges)
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


def generate_ground_texture(texture_type, size_px):
    """
    Génère un grand patch de texture de sol (carré size_px x size_px) selon le type demandé.
    Toutes les couleurs sont en BGR (convention OpenCV). Ce patch est ensuite plaqué au sol
    via une véritable homographie 3D (voir compute_ground_homography / warp_ground_texture),
    exactement comme le marqueur, pour que les lattes/brins/dalles convergent vers le même
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


def apply_lens_distortion(image, k1):
    h, w = image.shape[:2]
    f = max(h, w)
    cx, cy = w / 2.0, h / 2.0
    
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    x = (grid_x - cx) / f
    y = (grid_y - cy) / f
    r2 = x**2 + y**2
    
    distortion = 1.0 + k1 * r2
    
    map_x = (x * distortion * f + cx).astype(np.float32)
    map_y = (y * distortion * f + cy).astype(np.float32)
    
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def apply_chrominance_noise(image, luma_sigma_range, chroma_sigma_range, lowlight_boost):
    """
    Bruit typique des petits capteurs (IMX708) : plus fort sur la chrominance (Cr/Cb) que
    sur la luminance (Y), et amplifié dans les zones sombres (bruit de basse lumière).
    """
    luma_sigma = random.uniform(*luma_sigma_range)
    chroma_sigma = random.uniform(*chroma_sigma_range)

    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    luma = ycc[:, :, 0]
    lowlight_factor = 1.0 + (1.0 - luma / 255.0) * lowlight_boost

    ycc[:, :, 0] += np.random.normal(0, luma_sigma, luma.shape)
    ycc[:, :, 1] += np.random.normal(0, chroma_sigma, luma.shape) * lowlight_factor
    ycc[:, :, 2] += np.random.normal(0, chroma_sigma, luma.shape) * lowlight_factor

    ycc = np.clip(ycc, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


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


def composite_images(floor, shadow, marker):
    floor_f = floor.astype(np.float32)
    
    shadow_rgb = shadow[:, :, :3].astype(np.float32)
    shadow_alpha = np.expand_dims(shadow[:, :, 3].astype(np.float32) / 255.0, axis=2)
    bg_with_shadow = shadow_rgb * shadow_alpha + floor_f * (1.0 - shadow_alpha)
    
    marker_rgb = marker[:, :, :3].astype(np.float32)
    marker_alpha = np.expand_dims(marker[:, :, 3].astype(np.float32) / 255.0, axis=2)
    final = marker_rgb * marker_alpha + bg_with_shadow * (1.0 - marker_alpha)
    
    return np.clip(final, 0, 255).astype(np.uint8)


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

def process_single_marker(svg_path):
    """Fonction exécutée en parallèle pour générer les 500 images d'un marqueur."""
    # Sécurise l'aléa pour éviter les doublons de trajectoire entre processus
    random.seed(os.getpid() + int(time.time() * 1000) % 1000)
    np.random.seed(os.getpid() + int(time.time() * 1000) % 1000)
    
    marker_id = extract_marker_id(svg_path)
    marker_base = load_marker(svg_path, size=600)
    
    if marker_base is None:
        return 0, 0
        
    output_dir = CONFIG["output_dir"]
    frames_per_traj = CONFIG["frames_per_trajectory"]
    width, height = CONFIG["output_resolution"]
    fx, _, _, _ = get_camera_intrinsics(width, height)
    dt_frame = CONFIG["trajectory_duration_s"] / frames_per_traj
    
    local_detector = ArucoDetector()
    local_detected_count = 0
    
    traj = generate_flight_trajectory(CONFIG, marker_id=marker_id)
    trajectory_id = f"traj_{marker_id}_{int(random.random() * 1e6):06d}"
    imu = simulate_mpu6050_imu(traj, dt_frame, CONFIG)

    # --- Environnement de la trajectoire (fixe pendant les ~500 frames = ~1s de vol) ---
    texture_type = random.choice(CONFIG["ground_texture_types"])
    tex_size_px = int(CONFIG["ground_texture_size_m"] * CONFIG["ground_texels_per_meter"])
    ground_texture = generate_ground_texture(texture_type, tex_size_px)

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

    for img_idx in range(frames_per_traj):
        altitude = float(traj["altitude"][img_idx])
        roll = float(traj["roll"][img_idx])
        pitch = float(traj["pitch"][img_idx])
        yaw = float(traj["yaw"][img_idx])
        yaw_rate = float(traj["yaw_rate"][img_idx])
        speed = float(traj["forward_speed"][img_idx])
        drone_x = float(traj["pos_x"][img_idx])
        drone_y = float(traj["pos_y"][img_idx])

        marker_w, shadow_w, pts_img, ground_H = apply_drone_rotation(
            marker_base, width, height, altitude, roll, pitch, yaw,
            drone_x, drone_y, sun_az_deg, sun_elev_deg, CONFIG
        )

        # Sol entier plaqué avec la MÊME homographie que le marqueur (même point de fuite)
        scene = warp_ground_texture(ground_texture, ground_H, CONFIG["ground_texels_per_meter"], width, height)

        scene = composite_images(scene, shadow_w, marker_w)
        
        motion_dist_m = speed * CONFIG["exposure_time"]
        motion_blur_len = motion_dist_m * (fx / altitude)
        motion_angle = 90.0 + yaw_rate * CONFIG["exposure_time"] * 10.0 + random.uniform(-3, 3)
        scene = apply_motion_blur(scene, motion_blur_len, motion_angle)
        
        lateral_speed = speed * np.sin(np.radians(yaw))
        shutter_shift_m = lateral_speed * CONFIG["rolling_shutter_readout"]
        shutter_shift_px = shutter_shift_m * (fx / altitude)
        scene = apply_rolling_shutter(scene, shutter_shift_px)
        
        scene = apply_lens_distortion(scene, CONFIG["k1_distortion"])

        # Profondeur de champ : flou proportionnel à l'écart avec la distance de mise au point,
        # + éventuel "saut" de mise au point (hunting) sur une fenêtre de frames.
        defocus_m = abs(altitude - focus_distance_m)
        ksize = CONFIG["dof_blur_base_px"] + defocus_m * CONFIG["dof_blur_coeff_px_per_m"]
        if hunt_start <= img_idx < hunt_end:
            ksize += CONFIG["autofocus_hunt_extra_ksize"]
        ksize = min(ksize, CONFIG["dof_max_ksize"])
        scene = apply_focus_blur(scene, ksize)

        vignette_strength = random.uniform(*CONFIG["vignette_strength_range"])
        scene = apply_vignette(scene, vignette_strength)

        ae_gain = random.uniform(*CONFIG["ae_gain_range"])
        scene = apply_auto_exposure(scene, ae_gain)

        scene = apply_chrominance_noise(
            scene, CONFIG["noise_luma_sigma_range"], CONFIG["noise_chroma_sigma_range"],
            CONFIG["chroma_lowlight_boost"]
        )
        
        detected_markers = local_detector.detect(scene)
        is_detected = any(m["id"] == int(marker_id) for m in detected_markers)
        if is_detected:
            local_detected_count += 1
        
        marker_size_px = calculate_marker_size(pts_img)
        marker_fully_in_frame = bool(np.all(
            (pts_img[:, 0] >= 0) & (pts_img[:, 0] < width) &
            (pts_img[:, 1] >= 0) & (pts_img[:, 1] < height)
        ))
        
        filename_base = f"marker_{marker_id}_{img_idx:03d}"
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
                "autofocus_hunting": bool(hunt_start <= img_idx < hunt_end),
                "ae_gain": round(float(ae_gain), 3),
                "vignette_strength": round(float(vignette_strength), 3),
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
            
    return frames_per_traj, local_detected_count

# ==============================================================================
# PIPELINE PRINCIPAL MULTI-PROCESSUS
# ==============================================================================

# ==============================================================================
# ÉTAPE 2 : DATASET ROI POUR CNN (fusion de Prepare.py)
# ==============================================================================
# Repart des images "caméra" (frame complète 640x480) générées ci-dessus et produit
# un second dataset : des crops recentrés/agrandis autour du marqueur (128x128 par défaut),
# avec augmentation (jitter de position + zoom arrière aléatoire) pour simuler l'incertitude
# d'un premier stade de détection/tracking en amont du CNN de pose.

def create_augmented_roi_dataset(config):
    """
    Parcourt le dataset caméra (images brutes 640x480), extrait la ROI autour du marqueur
    avec une taille de découpe FIXE pour préserver l'information d'altitude (Z),
    applique des augmentations (jitter, échelle) et sauvegarde le tout dans cnn_roi_dataset/.
    """
    input_dir = config["output_dir"]
    output_dir = "./cnn_roi_dataset"
    os.makedirs(output_dir, exist_ok=True)

    camera_jsons = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not camera_jsons:
        print(f"❌ Aucune donnée brute trouvée dans '{input_dir}'.")
        return 0

    augmentation_factor = config["roi_augmentation_factor"]
    jitter_px = config["roi_jitter_px"]
    scale_lo, scale_hi = config["roi_scale_range"]
    roi_w, roi_h = config["roi_size"]

    # 🎯 CONFIGURATION DE LA MARGE FIXE 
    # Au lieu de s'adapté à la taille du marqueur, on découpe une zone fixe (ex: 180x180 pixels).
    # Si le drone est loin, le marqueur sera petit dans ce carré. S'il est proche, il sera grand.
    base_fixed_margin = 90  # Marge de 90px autour du centre = carré de 180x180 dans l'image 640x480
    
    roi_count = 0

    for json_path in camera_jsons:
        img_path = json_path.replace(".json", ".png")
        if not os.path.exists(img_path):
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        corners = data.get("aruco_corners", [])
        if not corners or len(corners) < 4:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img, _ = img.shape

        # Trouver le centre exact du marqueur à l'écran
        corners = np.array(corners)
        center_x = int(np.mean(corners[:, 0]))
        center_y = int(np.mean(corners[:, 1]))

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # ✨ Correction de la récupération des données : on reconstruit le vecteur cible
        # à partir des clés réelles présentes dans le dataset caméra brut.
        xyz_rpy = [
            data.get("x_m", 0.0),
            data.get("y_m", 0.0),
            data.get("distance_m", data.get("z_m", 0.0)), # Capture distance_m ou z_m
            data.get("roll_deg", 0.0),
            data.get("pitch_deg", 0.0),
            data.get("yaw_deg", 0.0)
        ]

        for i in range(augmentation_factor):
            if i == 0:
                dx, dy = 0, 0
                scale_modifier = 1.0
            else:
                dx = random.randint(-jitter_px, jitter_px)
                dy = random.randint(-jitter_px, jitter_px)
                scale_modifier = random.uniform(scale_lo, scale_hi)

            # Application de l'augmentation d'échelle sur notre boîte fixe
            current_margin = int(base_fixed_margin * scale_modifier)

            crop_xmin = max(0, center_x - current_margin + dx)
            crop_xmax = min(w_img, center_x + current_margin + dx)
            crop_ymin = max(0, center_y - current_margin + dy)
            crop_ymax = min(h_img, center_y + current_margin + dy)

            crop_w = crop_xmax - crop_xmin
            crop_h = crop_ymax - crop_ymin

            if crop_w <= 10 or crop_h <= 10:
                continue

            # Extraction et redimensionnement strict en 128x128
            roi_img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            roi_resized = cv2.resize(roi_img, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

            out_img_name = f"{base_name}_roi_{i}.png"
            out_json_name = f"{base_name}_roi_{i}.json"
            
            cv2.imwrite(os.path.join(output_dir, out_img_name), roi_resized)

            # Sauvegarde des métadonnées formatées avec la clé attendue par le CNN
            roi_data = {
                "marker_id": data["marker_id"],
                "target_pose_xyz_rpy": xyz_rpy,          # 🔥 Corrigé ici : injecte la structure [6]
                "aruco_corners": data["aruco_corners"]
            }
            with open(os.path.join(output_dir, out_json_name), 'w') as f_out:
                json.dump(roi_data, f_out, indent=4)

            roi_count += 1

    return roi_count


def main():
    parser = argparse.ArgumentParser(description="Ardu-Citron Dataset Generator (Caméra + ROI CNN)")
    parser.add_argument("--skip-camera", action="store_true", help="Ne pas régénérer le dataset caméra (frame complète)")
    parser.add_argument("--skip-roi", action="store_true", help="Ne pas générer le dataset ROI (crops CNN)")
    args = parser.parse_args()

    print("=== [Ardu-Citron Dataset Generator - Multi-processus 4 Coeurs] ===")

    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_camera:
        svg_files = glob.glob("4x4_1000-*.svg")
        if not svg_files:
            svg_files = glob.glob(os.path.join(CONFIG["markers_dir"], "4x4_1000-*.svg"))

        if not svg_files:
            print(f"\n[ERREUR] Aucun fichier '4x4_1000-*.svg' trouvé.")
            return

        svg_files = sorted(svg_files, key=extract_marker_id)
        num_markers = len(svg_files)
        frames_per_traj = CONFIG["frames_per_trajectory"]
        total_images = num_markers * frames_per_traj

        print(f"-> {num_markers} marqueurs détectés.")
        print(f"-> Allocation de 4 processus parallèles...")
        print(f"-> [1/2] Génération du dataset CAMÉRA : {total_images} images ({num_markers} marqueurs x {frames_per_traj} frames/trajectoire).")

        start_time = time.time()

        total_processed = 0
        total_detected = 0

        print_progress_bar(0, total_images, prefix='Progression:', suffix='Terminé', length=50)

        # Utilisation de ProcessPoolExecutor fixé à 4 workers maximum comme demandé
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_single_marker, svg_path): svg_path for svg_path in svg_files}

            for future in as_completed(futures):
                try:
                    processed, detected = future.result()
                    total_processed += processed
                    total_detected += detected
                    # Mise à jour globale immédiate
                    print_progress_bar(total_processed, total_images, prefix='Progression:', suffix='Terminé', length=50)
                except Exception as e:
                    svg_path = futures[future]
                    print(f"\n[ERREUR] Le processus pour {svg_path} a planté : {e}")

        elapsed_time = time.time() - start_time
        success_rate = (total_detected / total_images) * 100 if total_images > 0 else 0

        print(f"\n[SUCCÈS] Dataset CAMÉRA généré dans '{output_dir}/' !")
        print(f"⏱️ Temps d'exécution total : {elapsed_time:.2f} secondes (soit {total_images / elapsed_time:.1f} images/s)")
        print(f"🎯 Taux de détectabilité initial par ArucoDetector : {success_rate:.2f} % ({total_detected}/{total_images})")
    else:
        print("-> [1/2] Dataset caméra ignoré (--skip-camera).")

    if not args.skip_roi:
        expected_roi = None
        try:
            n_camera = len(glob.glob(os.path.join(output_dir, "*.png")))
            expected_roi = n_camera * CONFIG["roi_augmentation_factor"]
        except Exception:
            pass
        print(f"\n-> [2/2] Génération du dataset ROI (crops CNN {CONFIG['roi_size'][0]}x{CONFIG['roi_size'][1]})"
              + (f", ~{expected_roi} images attendues." if expected_roi else "."))
        roi_start = time.time()
        roi_count = create_augmented_roi_dataset(CONFIG)
        roi_elapsed = time.time() - roi_start
        print(f"⏱️ Temps ROI : {roi_elapsed:.2f} secondes ({roi_count / roi_elapsed:.1f} images/s)" if roi_elapsed > 0 else "")
    else:
        print("-> [2/2] Dataset ROI ignoré (--skip-roi).")

    print(f"\n📦 Dataset CAMÉRA : '{CONFIG['output_dir']}/'  |  📦 Dataset ROI : '{CONFIG['roi_output_dir']}/'")


if __name__ == "__main__":
    # Indispensable pour la portabilité cross-platform des sous-processus
    multiprocessing.freeze_support()
    main()
