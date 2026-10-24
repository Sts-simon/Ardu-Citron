#!/usr/bin/env python3
"""CM4 camera/UART runtime for pose telemetry and gated v1 guidance setpoints.

Configure real IMU/GPS providers and validate the aircraft on the ground first.
ESP32 AUTO remains separately disabled by default and requires the physical
mode/arm inputs plus fresh RC and navigation data.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import cv2

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOL = ROOT / "Sol"
sys.path.insert(0, str(SOL))

from autonomy import AutopilotGuidance, GpsFix, ImuSample, Marker, MarkerLocalizer, NavigationEstimator
from vision_uart import GuidanceCommand, VisionPose, encode_guidance, encode_pose
import aruco_dedup_v2 as aruco


def load_config(path: pathlib.Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    camera = config["camera"]
    fx, fy = camera["fx_px"], camera.get("fy_px", camera["fx_px"])
    matrix = [[fx, 0.0, camera["cx_px"]], [0.0, fy, camera["cy_px"]], [0.0, 0.0, 1.0]]
    markers = [Marker(int(m["id"]), float(m["north_m"]), float(m["east_m"]),
                      float(m["altitude_m"]), float(m["size_m"]), float(m.get("yaw_deg", 0.0)))
               for m in config["markers"]]
    camera_cfg = config["camera"]
    config["_localizer"] = MarkerLocalizer(
        matrix, camera_cfg.get("distortion", [0, 0, 0, 0, 0]), markers,
        camera_cfg.get("camera_to_body"), camera_cfg.get("camera_position_body_m", [0, 0, 0]),
        camera_cfg.get("max_reprojection_error_px", 3.0))
    return config


def run(config: dict, camera_index: int, dry_run: bool = False) -> int:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra index {camera_index}")
    detector = aruco.ArucoCleanDetector(dict_name=config["aruco_dictionary"], preset="balanced",
                                        stabilize=False, roi_tracking=True, full_scan_every=8)
    estimator = NavigationEstimator(config["navigation"])
    guidance = AutopilotGuidance(config["autopilot"])
    uart = None
    if not dry_run:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial absent; installer les requirements Pi") from exc
        uart_cfg = config["uart"]
        uart = serial.Serial(uart_cfg["device"], int(uart_cfg["baud"]), timeout=0,
                             write_timeout=float(uart_cfg.get("write_timeout_s", 0.02)))
    sequence = 0
    command_sequence = 0
    last_command = None
    command_refresh_s = float(config["autopilot"].get("command_refresh_s", 0.10))
    last_command_sent_s = float("-inf")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Lecture caméra interrompue")
            now = time.monotonic()
            detections, _ = detector.detect(frame, t=now)
            visual_pose = config["_localizer"].estimate(detections, now)
            imu_sample = config.get("imu_provider", lambda: None)()
            if imu_sample is not None:
                estimator.predict(imu_sample)
            if visual_pose is not None:
                estimator.update_vision(visual_pose)
            gps_fix = config.get("gps_provider", lambda: None)()
            if gps_fix is not None:
                estimator.update_gps(gps_fix)
            state = estimator.state(now)
            decision = guidance.update(state)
            print(json.dumps({
                "t": now, "nav_valid": state.valid, "nav_source": state.source,
                "position_ned_m": [state.north_m, state.east_m, -state.altitude_m],
                "position_sigma_m": state.position_sigma_m,
                "marker_ids": list(visual_pose.marker_ids) if visual_pose else [],
                "autopilot_status": decision.status, "guidance_reason": decision.reason,
                "target_marker_id": decision.target_marker_id,
                "setpoint": (None if decision.command is None else {
                    "roll_deg": decision.command.roll_deg, "pitch_deg": decision.command.pitch_deg,
                    "yaw_rate_dps": decision.command.yaw_rate_dps}),
            }, separators=(",", ":")), flush=True)
            if uart is not None:
                # Pose stays telemetry. Guidance has its own bounded, CRC-protected
                # setpoint message; ESP32 additionally requires the physical AUTO switch.
                if visual_pose is not None:
                    pose = VisionPose(sequence & 0xFFFF, int(now * 1000) & 0xFFFFFFFF,
                                      visual_pose.north_m, visual_pose.east_m, -visual_pose.altitude_m,
                                      visual_pose.roll_deg, visual_pose.pitch_deg, visual_pose.yaw_deg,
                                      visual_pose.quality)
                    uart.write(encode_pose(pose))
                    sequence += 1
                if config["autopilot"].get("enabled", False) and decision.command is not None and state.valid:
                    setpoint = (round(decision.command.roll_deg, 1), round(decision.command.pitch_deg, 1),
                                round(decision.command.yaw_rate_dps, 1), round(state.position_sigma_m, 1))
                    # Repeat fresh, changing setpoints often enough to honor the
                    # ESP32's short validity lease; never trust Pi/ESP clock offset.
                    if setpoint != last_command or now - last_command_sent_s >= command_refresh_s:
                        cmd = GuidanceCommand(command_sequence & 0xFFFF, int(now * 1000) & 0xFFFFFFFF,
                                              decision.command.roll_deg, decision.command.pitch_deg,
                                              decision.command.yaw_rate_dps, 250,
                                              float(max(0.0, min(1.0, 1.0 - state.position_sigma_m / 20.0))))
                        uart.write(encode_guidance(cmd))
                        command_sequence += 1
                        last_command = setpoint
                        last_command_sent_s = now
                else:
                    last_command = None
                    last_command_sent_s = float("-inf")
            elapsed = time.monotonic() - now
            time.sleep(max(0.0, 1.0 / float(config["runtime_hz"]) - elapsed))
    finally:
        cap.release()
        if uart is not None:
            uart.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=HERE / "autonomy.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="camera + localization only, no UART")
    args = parser.parse_args()
    return run(load_config(args.config), args.camera, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
