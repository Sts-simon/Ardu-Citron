#!/usr/bin/env python3
"""Shared Ardu-Citron navigation core for the CM4 runtime and software emulator.

This module computes navigation estimates and bounded attitude/rate setpoints. It
is not flight-certified and deliberately does not drive PWM or motor hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import cv2
import numpy as np

G_M_S2 = 9.80665


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _rotation_body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


@dataclass(frozen=True)
class Marker:
    marker_id: int
    north_m: float
    east_m: float
    altitude_m: float
    size_m: float
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class VisualPose:
    time_s: float
    north_m: float
    east_m: float
    altitude_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    quality: float
    reprojection_error_px: float
    marker_ids: tuple[int, ...]


@dataclass(frozen=True)
class ImuSample:
    time_s: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    specific_force_body_m_s2: tuple[float, float, float]


@dataclass(frozen=True)
class GpsFix:
    time_s: float
    north_m: float
    east_m: float
    altitude_m: float
    accuracy_m: float


@dataclass(frozen=True)
class NavigationState:
    time_s: float
    north_m: float
    east_m: float
    altitude_m: float
    velocity_north_m_s: float
    velocity_east_m_s: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    position_sigma_m: float
    valid: bool
    source: str

    @property
    def speed_m_s(self) -> float:
        return math.hypot(self.velocity_north_m_s, self.velocity_east_m_s)

    @property
    def course_rad(self) -> float:
        if self.speed_m_s < 0.1:
            return math.radians(self.yaw_deg)
        return math.atan2(self.velocity_east_m_s, self.velocity_north_m_s)


class MarkerLocalizer:
    """Recover camera/body pose in local NED coordinates from mapped ArUco corners."""

    def __init__(self, camera_matrix: Sequence[Sequence[float]], distortion: Sequence[float],
                 markers: Iterable[Marker], camera_to_body: Sequence[Sequence[float]] | None = None,
                 camera_position_body_m: Sequence[float] = (0.0, 0.0, 0.0),
                 max_reprojection_error_px: float = 3.0):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.distortion = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
        self.markers = {int(m.marker_id): m for m in markers}
        self.camera_to_body = np.asarray(camera_to_body if camera_to_body is not None else
                                         [[0, -math.sqrt(0.5), math.sqrt(0.5)],
                                          [1, 0, 0], [0, math.sqrt(0.5), math.sqrt(0.5)]],
                                         dtype=np.float64).reshape(3, 3)
        if not np.allclose(self.camera_to_body.T @ self.camera_to_body, np.eye(3), atol=1e-5):
            raise ValueError("camera_to_body must be an orthonormal 3x3 rotation")
        self.camera_position_body = np.asarray(camera_position_body_m, dtype=float).reshape(3)
        self.max_error = float(max_reprojection_error_px)
        if not self.markers or self.max_error <= 0:
            raise ValueError("marker map and positive reprojection threshold are required")

    @staticmethod
    def _world_corners(marker: Marker) -> np.ndarray:
        """Four corners in detector order (image TL, TR, BR, BL), world NED."""
        angle = math.radians(marker.yaw_deg)
        up = np.array([math.cos(angle), math.sin(angle), 0.0])
        right = np.array([-math.sin(angle), math.cos(angle), 0.0])
        center = np.array([marker.north_m, marker.east_m, -marker.altitude_m])
        half = marker.size_m / 2.0
        return np.array([center + half * up - half * right,
                         center + half * up + half * right,
                         center - half * up + half * right,
                         center - half * up - half * right], dtype=np.float64)

    def estimate(self, detections: Iterable[object], time_s: float) -> VisualPose | None:
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        marker_ids: list[int] = []
        for detection in detections:
            marker = self.markers.get(int(getattr(detection, "marker_id", -1)))
            corners = getattr(detection, "corners", None)
            if marker is None or corners is None:
                continue
            image = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            if not np.all(np.isfinite(image)):
                continue
            object_points.extend(self._world_corners(marker))
            image_points.extend(image)
            marker_ids.append(marker.marker_id)
        if not object_points:
            return None

        obj = np.asarray(object_points, dtype=np.float64)
        img = np.asarray(image_points, dtype=np.float64)
        # IPPE handles one planar marker (four points); use all visible mapped
        # corners together so multiple markers reduce pose ambiguity and noise.
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, self.camera_matrix, self.distortion,
                                          flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return None
        if not ok:
            return None
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.camera_matrix, self.distortion,
                                              rvec, tvec)
        except (cv2.error, AttributeError):
            pass
        projected, _ = cv2.projectPoints(obj, rvec, tvec, self.camera_matrix, self.distortion)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img) ** 2, axis=1))))
        camera_points = (cv2.Rodrigues(rvec)[0] @ obj.T + tvec.reshape(3, 1)).T
        if not np.all(np.isfinite(camera_points)) or np.any(camera_points[:, 2] <= 0.0):
            return None
        if not math.isfinite(error) or error > self.max_error:
            return None

        world_to_camera = cv2.Rodrigues(rvec)[0]
        camera_to_world = world_to_camera.T
        camera_position_ned = -camera_to_world @ tvec.reshape(3)
        body_to_world = camera_to_world @ self.camera_to_body.T
        body_position_ned = camera_position_ned - body_to_world @ self.camera_position_body
        yaw = math.atan2(body_to_world[1, 0], body_to_world[0, 0])
        pitch = math.asin(float(np.clip(-body_to_world[2, 0], -1.0, 1.0)))
        roll = math.atan2(body_to_world[2, 1], body_to_world[2, 2])
        quality = float(np.clip(math.exp(-error / self.max_error) * min(1.0, len(marker_ids) / 2.0), 0.0, 1.0))
        return VisualPose(float(time_s), float(body_position_ned[0]), float(body_position_ned[1]),
                          float(-body_position_ned[2]), math.degrees(roll), math.degrees(pitch),
                          math.degrees(yaw), quality, error, tuple(sorted(set(marker_ids))))


class NavigationEstimator:
    """Small complementary INS corrected by mapped-marker poses and optional GPS."""

    def __init__(self, config: dict):
        self.use_imu = bool(config.get("use_imu", True))
        self.use_vision = bool(config.get("use_vision", True))
        self.use_gps = bool(config.get("use_gps", False))
        self.fix_timeout_s = float(config.get("fix_timeout_s", 2.0))
        self.process_noise_m_s = float(config.get("process_noise_m_s", 0.8))
        self.vision_gain = float(config.get("vision_gain", 0.65))
        self.gps_max_accuracy_m = float(config.get("gps_max_accuracy_m", 8.0))
        if self.fix_timeout_s <= 0 or not 0 < self.vision_gain <= 1:
            raise ValueError("invalid navigation timeout or vision gain")
        self.n = self.e = self.alt = 0.0
        self.vn = self.ve = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.sigma = math.inf
        self.last_imu_s: float | None = None
        self.last_fix_s: float | None = None
        self.last_imu_s_seen: float | None = None
        self.source = "uninitialized"

    def predict(self, sample: ImuSample) -> None:
        t = float(sample.time_s)
        if not self.use_imu:
            return
        self.last_imu_s_seen = t
        if self.last_imu_s is None:
            self.last_imu_s = t
            self.roll, self.pitch, self.yaw = map(math.radians,
                                                  (sample.roll_deg, sample.pitch_deg, sample.yaw_deg))
            return
        dt = t - self.last_imu_s
        self.last_imu_s = t
        if not 0.0 < dt <= 0.25:
            self.sigma = min(1000.0, self.sigma + self.process_noise_m_s * max(dt, 0.0))
            return
        self.roll, self.pitch, self.yaw = map(math.radians,
                                              (sample.roll_deg, sample.pitch_deg, sample.yaw_deg))
        specific = np.asarray(sample.specific_force_body_m_s2, dtype=float).reshape(3)
        if not np.all(np.isfinite(specific)):
            self.sigma += self.process_noise_m_s * dt
            return
        accel_ned = _rotation_body_to_ned(self.roll, self.pitch, self.yaw) @ specific
        accel_ned[2] += G_M_S2
        accel_ned[:2] = np.clip(accel_ned[:2], -20.0, 20.0)
        self.n += self.vn * dt + 0.5 * accel_ned[0] * dt * dt
        self.e += self.ve * dt + 0.5 * accel_ned[1] * dt * dt
        self.vn += accel_ned[0] * dt
        self.ve += accel_ned[1] * dt
        self.sigma = min(1000.0, self.sigma + self.process_noise_m_s * dt)
        if self.source == "uninitialized":
            self.source = "imu_dead_reckoning"

    def update_vision(self, pose: VisualPose) -> bool:
        if not self.use_vision or pose.quality <= 0.0 or pose.quality > 1.0:
            return False
        measured = np.array([pose.north_m, pose.east_m, pose.altitude_m], dtype=float)
        if not np.all(np.isfinite(measured)):
            return False
        if self.last_fix_s is not None and pose.time_s < self.last_fix_s:
            return False
        if self.last_fix_s is not None and np.isfinite(self.sigma):
            jump = math.hypot(pose.north_m - self.n, pose.east_m - self.e)
            if jump > max(35.0, 5.0 * self.sigma):
                return False
        gain = float(np.clip(self.vision_gain * pose.quality, 0.15, 0.95))
        if self.last_fix_s is None:
            self.n, self.e, self.alt = measured
            self.vn = self.ve = 0.0
        else:
            dt = max(0.02, pose.time_s - self.last_fix_s)
            dn, de = pose.north_m - self.n, pose.east_m - self.e
            self.n += gain * dn
            self.e += gain * de
            self.alt += gain * (pose.altitude_m - self.alt)
            if self.use_imu and dt <= self.fix_timeout_s:
                beta = min(0.35, gain * 0.35)
                self.vn += beta * dn / dt
                self.ve += beta * de / dt
        self.roll, self.pitch, self.yaw = map(math.radians,
                                              (pose.roll_deg, pose.pitch_deg, pose.yaw_deg))
        self.sigma = max(0.25, self.sigma * (1.0 - gain) if np.isfinite(self.sigma) else 1.0)
        self.last_fix_s = float(pose.time_s)
        self.source = "vision+imu" if self.use_imu else "vision"
        return True

    def update_gps(self, fix: GpsFix) -> bool:
        if not self.use_gps or not 0.0 < fix.accuracy_m <= self.gps_max_accuracy_m:
            return False
        if not all(math.isfinite(v) for v in (fix.north_m, fix.east_m, fix.altitude_m, fix.time_s)):
            return False
        if self.last_fix_s is not None and fix.time_s < self.last_fix_s:
            return False
        gain = float(np.clip(1.0 - fix.accuracy_m / self.gps_max_accuracy_m, 0.15, 0.8))
        if self.last_fix_s is None:
            self.n, self.e, self.alt = fix.north_m, fix.east_m, fix.altitude_m
            self.vn = self.ve = 0.0
        else:
            self.n += gain * (fix.north_m - self.n)
            self.e += gain * (fix.east_m - self.e)
            self.alt += gain * (fix.altitude_m - self.alt)
        self.sigma = min(self.sigma, max(1.0, fix.accuracy_m))
        self.last_fix_s = float(fix.time_s)
        self.source = "gps+imu" if self.use_imu else "gps"
        return True

    def state(self, now_s: float) -> NavigationState:
        fresh_fix = self.last_fix_s is not None and 0 <= now_s - self.last_fix_s <= self.fix_timeout_s
        imu_ok = not self.use_imu or (self.last_imu_s_seen is not None and
                                      0 <= now_s - self.last_imu_s_seen <= 0.5)
        valid = fresh_fix and imu_ok and self.sigma < 20.0
        return NavigationState(float(now_s), self.n, self.e, self.alt, self.vn, self.ve,
                               math.degrees(self.roll), math.degrees(self.pitch), math.degrees(self.yaw),
                               float(self.sigma), valid, self.source)


@dataclass(frozen=True)
class PilotCommand:
    roll_deg: float
    pitch_deg: float
    yaw_rate_dps: float


@dataclass(frozen=True)
class GuidanceDecision:
    command: PilotCommand | None
    status: str
    target_marker_id: int | None
    distance_to_fence_m: float
    reason: str


class AutopilotGuidance:
    """Cycles over mapped markers while steering toward the safe part of a box fence."""

    def __init__(self, config: dict):
        zone = config["zone"]
        self.n_min, self.n_max = float(zone["north_min_m"]), float(zone["north_max_m"])
        self.e_min, self.e_max = float(zone["east_min_m"]), float(zone["east_max_m"])
        self.margin = float(zone["margin_m"])
        if self.n_min >= self.n_max or self.e_min >= self.e_max or self.margin < 0:
            raise ValueError("invalid rectangular flight zone")
        if 2 * self.margin >= min(self.n_max - self.n_min, self.e_max - self.e_min):
            raise ValueError("fence margin leaves no safe interior")
        self.markers = tuple(config["markers"])
        if not self.markers:
            raise ValueError("at least one mapped overflight marker is required")
        for marker in self.markers:
            if not self._inside(float(marker["north_m"]), float(marker["east_m"]), self.margin):
                raise ValueError(f"marker {marker['id']} is outside the safe inset rectangle")
        self.max_bank = math.radians(float(config.get("max_bank_deg", 25.0)))
        self.max_yaw_rate = math.radians(float(config.get("max_yaw_rate_dps", 25.0)))
        self.lookahead_s = float(config.get("lookahead_s", 2.5))
        self.waypoint_radius = float(config.get("waypoint_radius_m", 4.0))
        self.target_alt = float(config.get("target_altitude_m", 6.0))
        self.altitude_gain = float(config.get("altitude_gain_deg_per_m", 0.4))
        self.index = 0
        self.last_distance_to_fence = math.inf

    def _inside(self, north: float, east: float, inset: float = 0.0) -> bool:
        return (self.n_min + inset <= north <= self.n_max - inset and
                self.e_min + inset <= east <= self.e_max - inset)

    def _distance_to_fence(self, n: float, e: float) -> float:
        return min(n - self.n_min, self.n_max - n, e - self.e_min, self.e_max - e)

    def update(self, state: NavigationState) -> GuidanceDecision:
        if not state.valid or not all(math.isfinite(v) for v in
                                      (state.north_m, state.east_m, state.speed_m_s, state.yaw_deg)):
            return GuidanceDecision(None, "NAV_UNCERTAIN", None, math.nan,
                                    "position fix stale or uncertainty too high")
        n, e = state.north_m, state.east_m
        self.last_distance_to_fence = self._distance_to_fence(n, e)
        center_n, center_e = (self.n_min + self.n_max) / 2, (self.e_min + self.e_max) / 2
        vn, ve = state.velocity_north_m_s, state.velocity_east_m_s
        speed = max(1.0, state.speed_m_s)
        projected_n = n + vn * self.lookahead_s
        projected_e = e + ve * self.lookahead_s
        turn_radius = speed * speed / (G_M_S2 * max(math.tan(self.max_bank), 0.05))
        fence_urgent = (not self._inside(n, e, self.margin) or
                        not self._inside(projected_n, projected_e, self.margin + min(turn_radius, self.margin)) or
                        self._distance_to_fence(n, e) <= self.margin + min(turn_radius, self.margin))
        if fence_urgent:
            target_n, target_e = center_n, center_e
            target_id = None
            reason = "FENCE_RECOVERY"
        else:
            target = self.markers[self.index]
            target_n, target_e = float(target["north_m"]), float(target["east_m"])
            target_id = int(target["id"])
            distance = math.hypot(target_n - n, target_e - e)
            if distance <= self.waypoint_radius:
                self.index = (self.index + 1) % len(self.markers)
                target = self.markers[self.index]
                target_n, target_e = float(target["north_m"]), float(target["east_m"])
                target_id = int(target["id"])
            reason = "MARKER_ROUTE"

        desired_course = math.atan2(target_e - e, target_n - n)
        course_error = wrap_pi(desired_course - state.course_rad)
        # L1-style lateral acceleration, capped to the bank limit. The ESP32
        # attitude/rate loops remain the only layer that drives control surfaces.
        lookahead_distance = max(15.0, speed * self.lookahead_s)
        desired_yaw_rate = float(np.clip(2.0 * speed / lookahead_distance * math.sin(course_error),
                                         -self.max_yaw_rate, self.max_yaw_rate))
        bank = math.atan2(speed * desired_yaw_rate, G_M_S2)
        bank = float(np.clip(bank, -self.max_bank, self.max_bank))
        pitch = float(np.clip((self.target_alt - state.altitude_m) * self.altitude_gain, -5.0, 5.0))
        command = PilotCommand(math.degrees(bank), pitch, math.degrees(desired_yaw_rate))
        return GuidanceDecision(command, "AUTO", target_id, self.last_distance_to_fence, reason)
