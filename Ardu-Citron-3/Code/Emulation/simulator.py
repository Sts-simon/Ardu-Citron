#!/usr/bin/env python3
"""Integrated software-in-the-loop glider emulation; no physical outputs."""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import pathlib
import subprocess
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
ESP32 = ROOT.parent / "ESP_32"
PI_DIR = ROOT.parent / "Pi"
SOL_DIR = ROOT.parent.parent / "Sol"
sys.path.insert(0, str(PI_DIR))
sys.path.insert(0, str(SOL_DIR))

from camera import VirtualCamera
from physics import G, Glider6DOF, rotation_body_to_ned
from vision_uart import (GuidanceCommand, GuidanceStreamDecoder, StreamDecoder, VisionPose,
                         encode_guidance, encode_pose)
from autonomy import (AutopilotGuidance, GpsFix, ImuSample, Marker, MarkerLocalizer,
                      NavigationEstimator)
import aruco_dedup_v2 as aruco

MODE_NAMES = {0: "STARTUP", 1: "DISARMED", 2: "MANUAL", 3: "STABILIZE", 4: "AUTO", 5: "FAILSAFE", 6: "SENSOR_FAULT"}
CSV_FIELDS = [
    "time_s", "mode", "armed", "grounded", "imu_valid", "rc_valid", "vision_detected",
    "nav_valid", "nav_source", "nav_north_m", "nav_east_m", "nav_sigma_m",
    "autopilot_status", "guidance_reason", "target_marker_id", "guidance_roll_deg",
    "guidance_pitch_deg", "guidance_yaw_rate_dps", "fence_distance_m",
    "detected_marker_count", "detected_marker_ids", "vision_quality", "vision_x_px", "vision_y_px",
    "position_n_m", "position_e_m", "altitude_m", "velocity_n_m_s", "velocity_e_m_s",
    "velocity_d_m_s", "airspeed_m_s", "alpha_deg", "beta_deg", "roll_deg", "pitch_deg", "yaw_deg",
    "p_dps", "q_dps", "r_dps", "specific_force_x_m_s2", "specific_force_y_m_s2", "specific_force_z_m_s2",
    "imu_gyro_x_dps", "imu_gyro_y_dps", "imu_gyro_z_dps", "imu_accel_x_g", "imu_accel_y_g", "imu_accel_z_g",
    "rc_roll_deg", "rc_pitch_deg", "rc_yaw_rate_dps", "pid_roll_output_deg", "pid_pitch_output_deg",
    "pid_yaw_output_deg", "servo_1_pulse_us", "servo_2_pulse_us", "servo_3_pulse_us",
    "servo_1_deflection_deg", "servo_2_deflection_deg", "servo_3_deflection_deg",
    "lift_n", "drag_n", "dynamic_pressure_pa", "force_x_n", "force_y_n", "force_z_n",
    "moment_l_nm", "moment_m_nm", "moment_n_nm", "uart_crc_valid", "uart_sequence", "imu_consecutive_errors",
]


def load_bridge(config: dict) -> ctypes.CDLL:
    build_dir = pathlib.Path(config["simulation"].get("build_dir", ROOT / ".build"))
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / "libsim_controller.so"
    sources = [ESP32 / "main" / file for file in (
        "kalman.cpp", "pid.cpp", "mixer.cpp", "servo_output.cpp", "imu_health.cpp",
        "flight_mode.cpp", "protocol.cpp")]
    command = ["g++", "-std=c++17", "-O2", "-fPIC", "-shared", "-Wall", "-Wextra",
               "-I" + str(ESP32 / "test/native"), "-I" + str(ESP32 / "main"),
               str(ROOT / "sim_bridge.cpp"), *(str(path) for path in sources), "-o", str(library)]
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(library))
    lib.sim_controller_create.restype = ctypes.c_void_p
    lib.sim_controller_destroy.argtypes = [ctypes.c_void_p]
    lib.sim_controller_initialize.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_uint32]
    lib.sim_controller_configure_auto.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
    lib.sim_controller_step.argtypes = [
        ctypes.c_void_p, ctypes.c_float, ctypes.c_uint32,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
        ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    return lib


def scenario_at(scenarios: list[dict], t: float, previous: tuple[float, float, float]) -> tuple[float, float, float, bool, bool]:
    command = previous
    arm = False
    disarm = False
    for scenario in scenarios:
        if t + 1e-9 < float(scenario["time_s"]):
            break
        if scenario["type"] == "rc_command":
            command = (float(scenario["roll_deg"]), float(scenario["pitch_deg"]), float(scenario["yaw_rate_dps"]))
            arm = bool(scenario.get("arm", False))
            disarm = bool(scenario.get("disarm", False))
    return (*command, arm, disarm)


def run(config_path: pathlib.Path, out_dir: pathlib.Path, duration_override: float | None = None) -> tuple[pathlib.Path, pathlib.Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    simulation = config["simulation"]
    rng = np.random.default_rng(int(simulation["seed"]))
    dt = float(simulation["dt_s"])
    duration = float(duration_override if duration_override is not None else simulation["duration_s"])
    out_dir.mkdir(parents=True, exist_ok=True)
    physics = Glider6DOF(config)
    camera = VirtualCamera(config, rng)
    detector = aruco.ArucoCleanDetector(dict_name=config["camera"]["marker_dictionary"], preset="balanced",
                                        stabilize=False, roi_tracking=True, full_scan_every=8)
    uart = StreamDecoder()
    guidance_decoder = GuidanceStreamDecoder()

    nav_config = config.get("navigation", {"use_imu": True, "use_vision": True, "use_gps": False})
    nav_estimator = NavigationEstimator(nav_config)
    zone = config.get("zone")
    autonomy_config = config.get("autopilot")
    guidance = None
    localizer = None
    if zone is not None and autonomy_config is not None:
        camera_cfg = config["camera"]
        fx = camera.width / (2.0 * math.tan(math.radians(camera_cfg["horizontal_fov_deg"]) / 2.0))
        matrix = [[fx, 0.0, camera.width / 2], [0.0, fx, camera.height / 2], [0.0, 0.0, 1.0]]
        markers = [Marker(int(m["id"]), float(m["north_m"]), float(m["east_m"]),
                          float(m.get("altitude_m", 0.0)), float(m["size_m"]), float(m.get("yaw_deg", 0.0)))
                   for m in config.get("markers", [])]
        camera_to_body = camera.camera_to_body if hasattr(camera, "camera_to_body") else None
        localizer = MarkerLocalizer(matrix, camera_cfg.get("distortion", [0, 0, 0, 0, 0]), markers,
                                    camera_to_body, camera_cfg.get("camera_position_body_m", [0, 0, 0]),
                                    camera_cfg.get("max_reprojection_error_px", 3.0))
        guidance = AutopilotGuidance({"zone": zone, "markers": config.get("markers", []), **autonomy_config})
    bridge = load_bridge(config)
    controller = bridge.sim_controller_create()
    try:
        initial_image, _ = camera.capture(physics.state, 0.0)
        initial_att = np.degrees(physics.state.attitude_rad)
        if not bridge.sim_controller_initialize(controller, ctypes.c_float(initial_att[0]),
                                                ctypes.c_float(initial_att[1]), 0):
            raise RuntimeError("ESP32 native controller failed startup")
        bridge.sim_controller_configure_auto(
            controller, int(bool(autonomy_config.get("enabled", False)) if autonomy_config else False),
            ctypes.c_float(float(autonomy_config.get("minimum_navigation_quality", 0.60))
                           if autonomy_config else 0.60))

        times = np.arange(0.0, duration + dt * 0.5, dt)
        accel_interval_us = max(1000, round(float(simulation["imu_period_s"]) * 1_000_000))
        camera_interval = max(1, round(float(simulation["camera_period_s"]) / dt))
        imu_interval = max(1, round(float(simulation["imu_period_s"]) / dt))
        control_interval = max(1, round(float(simulation["control_period_s"]) / dt))
        rc_interval = max(1, round(float(simulation.get("rc_period_s", 0.02)) / dt))
        servo_interval = max(1, round(float(simulation["servo_period_s"]) / dt))
        scenarios = sorted(config.get("scenarios", []), key=lambda item: item["time_s"])
        previous_command = (0.0, 0.0, 0.0)
        last_output = np.array([*np.degrees(physics.state.attitude_rad), 0.0, 0.0, 0.0, 1500, 1500, 1500, 1, 0, 0, 0], dtype=float)
        last_truth = []
        current_detections = []
        uart_crc_valid = guidance_crc_valid = 0
        uart_sequence = 0
        last_rc_valid = previous_rc_valid = False
        consecutive_imu_errors = 0
        last_arm = last_disarm = False
        last_imu_valid = last_imu_sample_valid = True
        last_measured_rates = np.zeros(3)
        last_accel = np.array([0.0, 0.0, 1.0])
        last_camera_frame = initial_image
        last_guidance = None
        guidance_sequence = 0
        last_nav_state = nav_estimator.state(0.0)
        logs: list[dict] = []

        for index, t in enumerate(times):
            t = float(t)
            state = physics.state
            _, _, _, _, detail = physics.derivatives(state.servo_pulses_us)
            body_to_ned = rotation_body_to_ned(*state.attitude_rad)
            gravity_ned = np.array([0.0, 0.0, physics.gravity])
            inertial_accel_ned = body_to_ned @ detail["accel_body_m_s2"] + gravity_ned
            ideal_accel_g = body_to_ned.T @ (inertial_accel_ned - gravity_ned) / G
            gyro_dps = np.degrees(state.rates_body_rad_s)
            imu_updated = index % imu_interval == 0
            if imu_updated:
                last_imu_sample_valid = rng.random() >= float(config["sensors"]["imu_dropout_probability"])
                if last_imu_sample_valid:
                    consecutive_imu_errors = 0
                    bias = np.asarray(config["sensors"]["gyro_bias_dps"])
                    walk = float(config["sensors"]["gyro_bias_walk_std_dps_sqrt_s"]) * math.sqrt(dt) * rng.normal(size=3)
                    config["sensors"]["gyro_bias_dps"] = (bias + walk).tolist()
                    last_measured_rates = gyro_dps + bias + rng.normal(0.0, config["sensors"]["gyro_noise_std_dps"], 3)
                    last_accel = ideal_accel_g + rng.normal(0.0, config["sensors"]["accel_noise_std_g"], 3)
                else:
                    consecutive_imu_errors += 1
            last_imu_valid = last_imu_sample_valid
            nav_estimator.predict(ImuSample(t, math.degrees(state.attitude_rad[0]),
                                            math.degrees(state.attitude_rad[1]), math.degrees(state.attitude_rad[2]),
                                            tuple(map(float, detail["accel_body_m_s2"]))))
            gps_cfg = config.get("gps_simulation", {})
            gps_period = float(gps_cfg.get("period_s", 1.0))
            if nav_config.get("use_gps", False) and gps_cfg.get("enabled", True) and index % max(1, round(gps_period / dt)) == 0:
                gps_position = state.position_ned_m + rng.normal(0.0, float(gps_cfg.get("position_noise_std_m", 1.5)), 3)
                nav_estimator.update_gps(GpsFix(t, float(gps_position[0]), float(gps_position[1]),
                                                float(-gps_position[2]), float(gps_cfg.get("accuracy_m", 3.0))))

            if index % camera_interval == 0:
                frame, last_truth = camera.capture(state, t)
                current_detections, _ = detector.detect(frame, t=t)
                last_camera_frame = frame
                if localizer is not None and nav_config.get("use_vision", True):
                    visual_pose = localizer.estimate(current_detections, t)
                    if visual_pose is not None:
                        nav_estimator.update_vision(visual_pose)
                for detection in current_detections:
                    matching = next((item for item in last_truth if item["marker_id"] == detection.marker_id), None)
                    if matching is None:
                        continue
                    quality = float(detector.scorer(frame, detection))
                    pose = VisionPose(uart_sequence & 0xFFFF, int(t * 1000),
                                      float(matching["center_px"][0]), float(matching["center_px"][1]),
                                      float(matching["distance_m"]), 0.0, 0.0, 0.0, float(np.clip(quality, 0.0, 1.0)))
                    uart_sequence += 1
                    encoded = bytearray(encode_pose(pose))
                    if rng.random() < float(config["sensors"]["uart_corruption_probability"]):
                        encoded[rng.integers(5, len(encoded))] ^= 0x40
                    decoded = uart.feed(bytes(encoded))
                    uart_crc_valid += len(decoded)

            rc_roll, rc_pitch, rc_yaw_rate, arm, disarm = scenario_at(scenarios, t, previous_command)
            previous_command = (rc_roll, rc_pitch, rc_yaw_rate)
            last_nav_state = nav_estimator.state(t)
            last_guidance = guidance.update(last_nav_state) if guidance is not None else None
            auto_enabled = bool(autonomy_config and autonomy_config.get("enabled", False))
            guidance_bytes = b""
            if auto_enabled and last_guidance and last_guidance.command and last_guidance.status == "AUTO" and last_nav_state.valid:
                command = GuidanceCommand(guidance_sequence & 0xFFFF, int(t * 1000) & 0xFFFFFFFF,
                                          last_guidance.command.roll_deg, last_guidance.command.pitch_deg,
                                          last_guidance.command.yaw_rate_dps, 250,
                                          float(np.clip(1.0 - last_nav_state.position_sigma_m / 20.0, 0.0, 1.0)))
                guidance_bytes = encode_guidance(command)
                guidance_sequence += 1
                guidance_decoder.feed(guidance_bytes)
                guidance_crc_valid += 1

            rc_updated = index % rc_interval == 0
            if rc_updated:
                previous_rc_valid = rng.random() >= float(config["sensors"]["rc_dropout_probability"])
                last_arm, last_disarm = arm, disarm
            last_rc_valid = previous_rc_valid
            accel_roll_deg = math.degrees(math.atan2(last_accel[1], last_accel[2]))
            accel_pitch_deg = math.degrees(math.atan2(-last_accel[0], math.hypot(last_accel[1], last_accel[2])))
            control_tick, servo_tick = index % control_interval == 0, index % servo_interval == 0
            if control_tick or servo_tick or imu_updated or rc_updated:
                values = (ctypes.c_float * 13)()
                now_us = int(t * 1_000_000)
                sample_us = now_us - (now_us % accel_interval_us)
                selected_mode = 2 if auto_enabled else 1
                guidance_buffer = (ctypes.c_uint8 * len(guidance_bytes)).from_buffer_copy(guidance_bytes) if guidance_bytes else None
                ok = bridge.sim_controller_step(
                    controller, ctypes.c_float(float(simulation["control_period_s"]) if control_tick else dt), int(t * 1000),
                    *[ctypes.c_float(v) for v in last_measured_rates], ctypes.c_float(accel_roll_deg), ctypes.c_float(accel_pitch_deg),
                    int(imu_updated), int(last_imu_sample_valid), int(rc_updated), int(last_rc_valid),
                    int(last_arm if rc_updated else False), int(last_disarm if rc_updated else False),
                    int(control_tick), int(servo_tick), ctypes.c_float(rc_roll), ctypes.c_float(rc_pitch),
                    ctypes.c_float(rc_yaw_rate), values, sample_us, selected_mode, guidance_buffer,
                    len(guidance_bytes), 0, 0.0, 0.0, 0.0)
                if not ok:
                    raise RuntimeError(f"native ESP32 controller step failed at t={t:.3f}s")
                last_output = np.array(list(values), dtype=float)

            state.servo_pulses_us = np.asarray(last_output[7:10], dtype=float)
            physics.step(state.servo_pulses_us, dt)
            _, _, _, velocity_ned, detail = physics.derivatives(state.servo_pulses_us)
            state = physics.state
            detected_ids = [d.marker_id for d in current_detections]
            best = current_detections[0] if current_detections else None
            quality = float(detector.scorer(last_camera_frame, best)) if best is not None else 0.0
            if (not np.all(np.isfinite(state.position_ned_m)) or not np.all(np.isfinite(state.velocity_body_m_s))
                    or not np.all(np.isfinite(state.attitude_rad)) or not np.all(np.isfinite(state.rates_body_rad_s))):
                raise FloatingPointError(f"Rigid-body state diverged at t={t:.6f}s")
            row = {
                "time_s": t, "mode": MODE_NAMES[int(last_output[10])], "armed": int(last_output[11] > 0.5),
                "grounded": int(state.grounded), "imu_valid": int(last_imu_valid), "rc_valid": int(last_rc_valid),
                "nav_valid": int(last_nav_state.valid), "nav_source": last_nav_state.source,
                "nav_north_m": last_nav_state.north_m, "nav_east_m": last_nav_state.east_m,
                "nav_sigma_m": last_nav_state.position_sigma_m,
                "autopilot_status": last_guidance.status if last_guidance else "DISABLED",
                "guidance_reason": last_guidance.reason if last_guidance else "not configured",
                "target_marker_id": last_guidance.target_marker_id if last_guidance else "",
                "guidance_roll_deg": last_guidance.command.roll_deg if last_guidance and last_guidance.command else 0.0,
                "guidance_pitch_deg": last_guidance.command.pitch_deg if last_guidance and last_guidance.command else 0.0,
                "guidance_yaw_rate_dps": last_guidance.command.yaw_rate_dps if last_guidance and last_guidance.command else 0.0,
                "fence_distance_m": last_guidance.distance_to_fence_m if last_guidance else math.nan,
                "vision_detected": int(best is not None), "detected_marker_count": len(current_detections),
                "detected_marker_ids": ";".join(map(str, detected_ids)), "vision_quality": quality,
                "vision_x_px": float(best.cx) if best else math.nan, "vision_y_px": float(best.cy) if best else math.nan,
                "position_n_m": state.position_ned_m[0], "position_e_m": state.position_ned_m[1],
                "altitude_m": -state.position_ned_m[2], "velocity_n_m_s": velocity_ned[0],
                "velocity_e_m_s": velocity_ned[1], "velocity_d_m_s": velocity_ned[2],
                "airspeed_m_s": detail["airspeed_m_s"], "alpha_deg": math.degrees(float(detail["alpha_rad"])),
                "beta_deg": math.degrees(float(detail["beta_rad"])), "roll_deg": math.degrees(state.attitude_rad[0]),
                "pitch_deg": math.degrees(state.attitude_rad[1]), "yaw_deg": math.degrees(state.attitude_rad[2]),
                "p_dps": math.degrees(state.rates_body_rad_s[0]), "q_dps": math.degrees(state.rates_body_rad_s[1]),
                "r_dps": math.degrees(state.rates_body_rad_s[2]),
                "specific_force_x_m_s2": detail["accel_body_m_s2"][0], "specific_force_y_m_s2": detail["accel_body_m_s2"][1],
                "specific_force_z_m_s2": detail["accel_body_m_s2"][2], "imu_gyro_x_dps": last_measured_rates[0],
                "imu_gyro_y_dps": last_measured_rates[1], "imu_gyro_z_dps": last_measured_rates[2],
                "imu_accel_x_g": last_accel[0], "imu_accel_y_g": last_accel[1], "imu_accel_z_g": last_accel[2],
                "rc_roll_deg": rc_roll, "rc_pitch_deg": rc_pitch, "rc_yaw_rate_dps": rc_yaw_rate,
                "pid_roll_output_deg": last_output[4], "pid_pitch_output_deg": last_output[5], "pid_yaw_output_deg": last_output[6],
                "servo_1_pulse_us": state.servo_pulses_us[0], "servo_2_pulse_us": state.servo_pulses_us[1],
                "servo_3_pulse_us": state.servo_pulses_us[2],
                "servo_1_deflection_deg": (state.servo_pulses_us[0] - 1500.0) / 50.0,
                "servo_2_deflection_deg": (state.servo_pulses_us[1] - 1500.0) / 50.0,
                "servo_3_deflection_deg": (state.servo_pulses_us[2] - 1500.0) / 50.0,
                "lift_n": detail["lift_n"], "drag_n": detail["drag_n"], "dynamic_pressure_pa": detail["dynamic_pressure_pa"],
                "force_x_n": detail["force_body_n"][0], "force_y_n": detail["force_body_n"][1], "force_z_n": detail["force_body_n"][2],
                "moment_l_nm": detail["moment_body_nm"][0], "moment_m_nm": detail["moment_body_nm"][1],
                "moment_n_nm": detail["moment_body_nm"][2], "uart_crc_valid": uart_crc_valid,
                "uart_sequence": uart_sequence & 0xFFFF, "guidance_crc_valid": guidance_crc_valid,
                "imu_consecutive_errors": consecutive_imu_errors,
            }
            logs.append(row)
            if state.grounded:
                for remaining in times[index + 1:]:
                    copied = row.copy(); copied["time_s"] = float(remaining); copied["grounded"] = 1
                    copied["armed"] = 0; copied["mode"] = "DISARMED"; logs.append(copied)
                break

        csv_path = out_dir / "flight_log.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS + ["guidance_crc_valid"])
            writer.writeheader(); writer.writerows(logs)
        plot_path = out_dir / "flight_summary.png"
        make_plots(logs, plot_path, camera.last_image)
        metadata = {
            "config": str(config_path), "seed": int(simulation["seed"]), "duration_s": duration,
            "steps": len(logs), "actual_aircraft_model": "6-DOF rigid body; quasi-steady coefficient aero",
            "motor_thrust_n": physics.thrust_n, "grounded": bool(physics.state.grounded),
            "autopilot_enabled": bool(autonomy_config and autonomy_config.get("enabled", False)),
            "uart_accepted_frames": uart_crc_valid, "guidance_accepted_frames": guidance_crc_valid,
            "disclaimer": "Provisional parameters; not flight-certified or a high-fidelity CFD model.",
        }
        (out_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return csv_path, plot_path
    finally:
        bridge.sim_controller_destroy(controller)


def make_plots(rows: list[dict], path: pathlib.Path, image: np.ndarray) -> None:
    time_s = np.array([row["time_s"] for row in rows])
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot([r["position_n_m"] for r in rows], [r["position_e_m"] for r in rows], label="trajectory")
    ax.set_title("Trajectoire au sol (N/E)"); ax.set_xlabel("Nord (m)"); ax.set_ylabel("Est (m)"); ax.axis("equal"); ax.grid(True)
    ax = axes[0, 1]
    for key in ("altitude_m", "airspeed_m_s"):
        ax.plot(time_s, [r[key] for r in rows], label=key)
    ax.set_title("Altitude et vitesse air"); ax.set_xlabel("Temps (s)"); ax.legend(); ax.grid(True)
    ax = axes[1, 0]
    for key in ("roll_deg", "pitch_deg", "yaw_deg"):
        ax.plot(time_s, [r[key] for r in rows], label=key)
    ax.set_title("Attitude"); ax.set_xlabel("Temps (s)"); ax.set_ylabel("Degrés"); ax.legend(); ax.grid(True)
    ax = axes[1, 1]
    for key in ("rc_roll_deg", "rc_pitch_deg", "pid_roll_output_deg", "pid_pitch_output_deg"):
        ax.plot(time_s, [r[key] for r in rows], label=key)
    ax.set_title("Consignes RC et sorties contrôleur"); ax.set_xlabel("Temps (s)"); ax.legend(fontsize=8); ax.grid(True)
    ax = axes[2, 0]
    for key in ("servo_1_pulse_us", "servo_2_pulse_us", "servo_3_pulse_us"):
        ax.plot(time_s, [r[key] for r in rows], label=key)
    ax.set_title("Sorties servos (µs)"); ax.set_xlabel("Temps (s)"); ax.legend(); ax.grid(True)
    ax = axes[2, 1]
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Dernière image caméra virtuelle"); ax.axis("off")
    fig.suptitle("Ardu-Citron — simulation logicielle provisoire (pas une validation de vol)", fontsize=14)
    fig.savefig(path, dpi=130); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "aircraft.json")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.out or ROOT / config["simulation"]["output_dir"]
    start = time.perf_counter()
    csv_path, plot_path = run(args.config, output, args.duration)
    print(f"CSV: {csv_path}"); print(f"Courbes: {plot_path}")
    print(f"Simulation terminée en {time.perf_counter() - start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
