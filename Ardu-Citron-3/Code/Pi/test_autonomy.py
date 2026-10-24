#!/usr/bin/env python3
import math
import pathlib
import sys
import unittest

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from autonomy import (AutopilotGuidance, GpsFix, ImuSample, Marker, MarkerLocalizer,
                      NavigationEstimator, NavigationState, VisualPose)


class Detection:
    def __init__(self, marker_id, corners):
        self.marker_id = marker_id
        self.corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)


def projected_detection(localizer, marker, camera_position, camera_to_body):
    world = localizer._world_corners(marker)
    body_to_world = camera_to_body
    camera_to_world = body_to_world @ localizer.camera_to_body
    world_to_camera = camera_to_world.T
    camera_points = (world - np.asarray(camera_position)) @ world_to_camera.T
    assert np.all(camera_points[:, 2] > 0)
    pixels, _ = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3),
                                  localizer.camera_matrix, localizer.distortion)
    return Detection(marker.marker_id, pixels.reshape(4, 2))


class AutonomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = __import__("json").loads((HERE / "autonomy.json").read_text())

    def test_marker_pose_recovers_known_local_position(self):
        marker = Marker(7, 0.0, 0.0, 0.0, 2.0, 0.0)
        intrinsic = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
        camera_to_body = np.array([[0, -math.sqrt(0.5), math.sqrt(0.5)],
                                   [1, 0, 0], [0, math.sqrt(0.5), math.sqrt(0.5)]])
        localizer = MarkerLocalizer(intrinsic, [0, 0, 0, 0, 0], [marker], camera_to_body)
        body_to_ned = np.eye(3)
        camera_position = np.array([0.0, 0.0, -10.0])
        detection = projected_detection(localizer, marker, camera_position, body_to_ned)
        pose = localizer.estimate([detection], 1.25)
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose.north_m, 0.0, delta=0.15)
        self.assertAlmostEqual(pose.east_m, 0.0, delta=0.15)
        self.assertAlmostEqual(pose.altitude_m, 10.0, delta=0.2)
        self.assertEqual(pose.marker_ids, (7,))

    def test_navigation_sources_are_switchable_and_stale_fixes_fail_closed(self):
        estimator = NavigationEstimator({"use_imu": True, "use_vision": True,
                                         "use_gps": False, "fix_timeout_s": 1.0})
        estimator.predict(ImuSample(0.0, 0, 0, 0, (0, 0, -9.80665)))
        self.assertTrue(estimator.update_vision(VisualPose(0, 10, 20, 6, 0, 0, 0, 1, 0.2, (7,))))
        self.assertTrue(estimator.state(0.5).valid)
        self.assertFalse(estimator.update_gps(GpsFix(0.5, 11, 20, 6, 2.0)))
        self.assertFalse(estimator.state(2.0).valid)
        no_imu = NavigationEstimator({"use_imu": False, "use_vision": True, "use_gps": True})
        self.assertTrue(no_imu.update_gps(GpsFix(1, 4, 5, 6, 1.0)))
        self.assertTrue(no_imu.state(1).valid)

    def test_guidance_routes_markers_and_recovers_inside_fence(self):
        cfg = __import__("json").loads((HERE / "autonomy.json").read_text())["autopilot"]
        cfg["zone"] = self.config["zone"]
        cfg["markers"] = self.config["markers"]
        guidance = AutopilotGuidance(cfg)
        center = NavigationState(1, 0, 0, 6, 10, 0, 0, 0, 0, 1, True, "vision+imu")
        decision = guidance.update(center)
        self.assertEqual(decision.status, "AUTO")
        self.assertEqual(decision.target_marker_id, 7)
        self.assertLessEqual(abs(decision.command.roll_deg), 25)
        near_edge = NavigationState(2, 85, 0, 6, 10, 0, 0, 0, 0, 1, True, "vision+imu")
        recovery = guidance.update(near_edge)
        self.assertEqual(recovery.reason, "FENCE_RECOVERY")
        self.assertIsNone(recovery.target_marker_id)
        self.assertLess(abs(recovery.command.yaw_rate_dps), 25.001)

    def test_untrusted_navigation_never_produces_guidance(self):
        guidance = AutopilotGuidance({**self.config["autopilot"],
                                      "zone": self.config["zone"], "markers": self.config["markers"]})
        state = NavigationState(4, 0, 0, 6, 10, 0, 0, 0, 0, 30, False, "imu_dead_reckoning")
        decision = guidance.update(state)
        self.assertEqual(decision.status, "NAV_UNCERTAIN")
        self.assertIsNone(decision.command)


if __name__ == "__main__":
    unittest.main()
