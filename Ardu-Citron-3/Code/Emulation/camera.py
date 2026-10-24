"""Synthetic forward/down camera tied to simulated aircraft pose."""
from __future__ import annotations

import math
import cv2
import numpy as np

from physics import rotation_body_to_ned


class VirtualCamera:
    def __init__(self, config: dict, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        camera = config["camera"]
        self.width = int(camera["width_px"])
        self.height = int(camera["height_px"])
        self.marker_size = float(camera["marker_size_m"])
        self.focal = self.width / (2.0 * math.tan(math.radians(camera["horizontal_fov_deg"]) / 2.0))
        self.cx, self.cy = self.width / 2.0, self.height / 2.0
        self.distortion = np.asarray(camera["distortion"], dtype=np.float64)
        intrinsic = np.array([[self.focal, 0.0, self.cx], [0.0, self.focal, self.cy], [0.0, 0.0, 1.0]])
        self.distort_map_x, self.distort_map_y = cv2.initUndistortRectifyMap(
            intrinsic, self.distortion, None, intrinsic, (self.width, self.height), cv2.CV_32FC1)
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, camera["marker_dictionary"]))
        self.markers = self._make_markers(camera["marker_ids"])
        self.last_visible: list[int] = []
        self.last_truth: list[dict] = []
        self.last_image = np.zeros((self.height, self.width), dtype=np.uint8)

    def _make_markers(self, ids: list[int]) -> list[tuple[int, np.ndarray]]:
        # NED world: ground at z=0. Layout at y=0, with known size and ID.
        markers = []
        camera = self.config["camera"]
        mapped = self.config.get("markers", [])
        if mapped:
            layout = [(int(item["id"]), float(item["north_m"]), float(item["east_m"]),
                       float(item.get("altitude_m", 0.0)), float(item.get("size_m", self.marker_size)),
                       float(item.get("yaw_deg", 0.0))) for item in mapped]
        else:
            layout = []
            for index, marker_id in enumerate(ids):
                center_n = float(camera.get("marker_layout_north_m", 8.0)) + index * float(camera.get("marker_spacing_m", 5.0))
                center_e = -float(camera.get("marker_lateral_spacing_m", 3.0)) + (index % 2) * 2.0 * float(camera.get("marker_lateral_spacing_m", 3.0))
                layout.append((int(marker_id), center_n, center_e, 0.0, self.marker_size, 0.0))
        for marker_id, center_n, center_e, altitude, s, yaw_deg in layout:
            yaw = math.radians(yaw_deg)
            forward = np.array([math.cos(yaw), math.sin(yaw)])
            right = np.array([-math.sin(yaw), math.cos(yaw)])
            center = np.array([center_n, center_e])
            corners_ne = np.array([center - forward * s/2 - right * s/2,
                                   center - forward * s/2 + right * s/2,
                                   center + forward * s/2 + right * s/2,
                                   center + forward * s/2 - right * s/2])
            s = float(s)
            points = np.column_stack((corners_ne[:, 0], corners_ne[:, 1],
                                      np.full(4, -float(altitude))))
            side = 256
            # Markers must have a quiet white region to be detectable on a
            # continuous scene; the generated bitmap itself contains only the
            # black marker tile, so add two white border modules before warp.
            tile = cv2.aruco.generateImageMarker(self.dictionary, int(marker_id), side)
            quiet = side // 4
            image = cv2.copyMakeBorder(tile, quiet, quiet, quiet, quiet,
                                       cv2.BORDER_CONSTANT, value=255)
            markers.append((int(marker_id), points, image))
        return markers

    def capture(self, state, time_s: float) -> tuple[np.ndarray, list[dict]]:
        camera = self.config["camera"]
        image = np.full((self.height, self.width), 155, dtype=np.uint8)
        body_to_ned = rotation_body_to_ned(*state.attitude_rad)
        # Camera optical axis points downward-forward by configurable mount angle.
        # Camera axes are right/down/forward; at 45 deg down it sees the ground
        # ahead of the glider and keeps horizon context for the vision pipeline.
        mount = math.radians(float(camera.get("mount_down_deg", 45.0)))
        camera_to_body = np.array([
            [0.0, -math.sin(mount), math.cos(mount)],
            [1.0, 0.0, 0.0],
            [0.0, math.cos(mount), math.sin(mount)],
        ], dtype=float)
        self.camera_to_body = camera_to_body
        camera_to_ned = body_to_ned @ camera_to_body
        world_to_camera = camera_to_ned.T
        cam_pos = state.position_ned_m
        visible: list[int] = []
        truth: list[dict] = []

        for marker_id, world_corners, marker_bitmap in self.markers:
            camera_xyz = (world_corners - cam_pos) @ world_to_camera.T
            if np.any(camera_xyz[:, 2] <= 0.15):
                continue
            intrinsic = np.array([[self.focal, 0.0, self.cx], [0.0, self.focal, self.cy], [0.0, 0.0, 1.0]])
            projected, _ = cv2.projectPoints(
                camera_xyz.astype(np.float64), np.zeros(3), np.zeros(3), intrinsic, self.distortion)
            projected = projected.reshape(4, 2).astype(np.float32)
            if float(cv2.contourArea(projected)) < 24.0:
                continue
            visible_fraction = _visible_fraction(projected, self.width, self.height)
            if visible_fraction < float(camera["minimum_visible_fraction"]):
                continue

            quiet = marker_bitmap.shape[0] // 6
            marker_side = marker_bitmap.shape[0] - 2 * quiet
            src = np.array([[quiet, quiet], [quiet + marker_side - 1, quiet],
                            [quiet + marker_side - 1, quiet + marker_side - 1],
                            [quiet, quiet + marker_side - 1]], dtype=np.float32)
            homography = cv2.getPerspectiveTransform(src, projected[[3, 2, 1, 0]])
            warped = cv2.warpPerspective(marker_bitmap, homography, (self.width, self.height),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=155)
            mask = cv2.warpPerspective(np.full_like(marker_bitmap, 255), homography,
                                       (self.width, self.height), flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            image[mask > 0] = warped[mask > 0]
            visible.append(marker_id)
            projected = projected[[3, 2, 1, 0]]
            noisy_corners = projected + self.rng.normal(0.0, camera["noise_sigma_px"], (4, 2)).astype(np.float32)
            truth.append({
                "marker_id": marker_id,
                "corners_px": noisy_corners,
                "center_px": noisy_corners.mean(axis=0),
                "distance_m": float(np.linalg.norm(world_corners.mean(axis=0) - cam_pos)),
                "visible_fraction": visible_fraction,
            })

        if self.rng.random() < float(camera["drop_probability"]):
            image.fill(155)
            visible = []
            truth = []
        exposure = float(camera["exposure_s"])
        readout = float(camera["readout_s"])
        blur_px = min(9, max(1, int(round(abs(state.rates_body_rad_s[1]) * exposure * self.focal))))
        if blur_px >= 3:
            if blur_px % 2 == 0: blur_px += 1
            image = cv2.GaussianBlur(image, (blur_px, blur_px), 0)
        row_shift = float(state.velocity_body_m_s[1] * readout * self.focal / max(1.0, -state.position_ned_m[2]))
        if abs(row_shift) > 0.2:
            map_x, map_y = np.meshgrid(np.arange(self.width, dtype=np.float32), np.arange(self.height, dtype=np.float32))
            map_x = map_x + (map_y / max(1, self.height - 1) - 0.5) * row_shift
            image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=155)
        image = np.clip(image.astype(float) + self.rng.normal(0.0, camera["noise_sigma_px"] * 5.0, image.shape), 0, 255).astype(np.uint8)
        self.last_visible = visible
        self.last_truth = truth
        self.last_image = image
        return image, truth


def _visible_fraction(points: np.ndarray, width: int, height: int) -> float:
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    area = max(1.0, float((x1 - x0) * (y1 - y0)))
    visible = max(0.0, min(width, x1) - max(0.0, x0)) * max(0.0, min(height, y1) - max(0.0, y0))
    return visible / area
