#!/usr/bin/env python3
import csv
import json
import pathlib
import tempfile
import unittest

import cv2
import numpy as np

import sys
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Pi"))
sys.path.insert(0, str(ROOT.parent.parent / "Sol"))
from camera import VirtualCamera
from physics import AircraftState, Glider6DOF
from simulator import run
from vision_uart import StreamDecoder, VisionPose, encode_pose
from autonomy import Marker, MarkerLocalizer, NavigationEstimator
import aruco_dedup_v2 as aruco


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "aircraft.json").read_text())

    def test_initial_aircraft_and_gravity(self):
        flight = Glider6DOF(self.config)
        speed, alpha, beta, _ = flight.air_data()
        self.assertAlmostEqual(speed, 10.0, places=5)
        self.assertGreater(flight.state.position_ned_m[2], -10.0)
        self.assertAlmostEqual(beta, 0.0, places=5)
        result = flight.step(flight.state.servo_pulses_us, 0.02)
        self.assertTrue(np.isfinite(result["lift_n"]))
        self.assertLess(flight.state.position_ned_m[2], 0.0)
        self.assertEqual(flight.thrust_n, 0.0)

    def test_camera_has_detectable_markers(self):
        camera = VirtualCamera(self.config, np.random.default_rng(2))
        state = AircraftState.from_config(self.config["aircraft"])
        frame, truth = camera.capture(state, 0.0)
        self.assertEqual(frame.shape, (480, 640))
        detector = aruco.ArucoCleanDetector(dict_name="DICT_4X4_50", roi_tracking=False)
        found, _ = detector.detect(frame, t=0.0)
        self.assertTrue(truth)
        self.assertTrue(any(item.marker_id in camera.last_visible for item in found))

    def test_emulator_camera_pose_uses_shared_pi_localizer(self):
        cfg = json.loads(json.dumps(self.config))
        camera = VirtualCamera(cfg, np.random.default_rng(2))
        state = AircraftState.from_config(cfg["aircraft"])
        frame, _ = camera.capture(state, 0.0)
        detector = aruco.ArucoCleanDetector(dict_name=cfg["camera"]["marker_dictionary"], roi_tracking=False)
        found, _ = detector.detect(frame, t=0.0)
        fx = camera.focal
        matrix = [[fx, 0.0, camera.cx], [0.0, fx, camera.cy], [0.0, 0.0, 1.0]]
        markers = [Marker(int(m["id"]), m["north_m"], m["east_m"], m["altitude_m"],
                          m["size_m"], m["yaw_deg"]) for m in cfg["markers"]]
        localizer = MarkerLocalizer(matrix, cfg["camera"]["distortion"], markers,
                                    camera.camera_to_body)
        pose = localizer.estimate(found, 0.0)
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose.altitude_m, -state.position_ned_m[2], delta=0.6)
        self.assertAlmostEqual(pose.north_m, state.position_ned_m[0], delta=0.6)

    def test_integrated_outputs_and_uart_roundtrip(self):
        # Avoid writing temporary files into the checked-in output directory.
        cfg = json.loads(json.dumps(self.config))
        cfg["simulation"]["duration_s"] = 0.6
        cfg["simulation"]["seed"] = 17
        with tempfile.TemporaryDirectory(prefix="ardu-citron-sim-test-") as temp:
            config_path = pathlib.Path(temp) / "aircraft.json"
            config_path.write_text(json.dumps(cfg))
            output = pathlib.Path(temp) / "output"
            csv_path, plot_path = run(config_path, output)
            self.assertTrue(csv_path.is_file() and plot_path.is_file())
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 10)
            self.assertIn("altitude_m", rows[0])
            self.assertIn("servo_1_pulse_us", rows[0])
            self.assertIn("autopilot_status", rows[0])
            self.assertIn("nav_valid", rows[0])
            self.assertTrue(all(row["mode"] in {"DISARMED", "STABILIZE", "FAILSAFE", "SENSOR_FAULT"} for row in rows))
            self.assertTrue(all(1000 <= float(item[f"servo_{i}_pulse_us"]) <= 2000
                                for item in rows for i in (1, 2, 3)))
            for field in ("altitude_m", "airspeed_m_s", "roll_deg", "pitch_deg", "p_dps", "q_dps"):
                values = [float(row[field]) for row in rows]
                self.assertTrue(all(np.isfinite(value) for value in values), field)
                self.assertLess(max(abs(value) for value in values), 100.0, field)
            self.assertGreater(float(rows[-1]["altitude_m"]), 0.0)
            self.assertTrue(all(row["autopilot_status"] in {"AUTO", "NAV_UNCERTAIN"} for row in rows))

        pose = VisionPose(4, 1000, 12.0, 33.0, 4.0, 0.0, 0.0, 0.0, 0.8)
        decoder = StreamDecoder()
        self.assertEqual(decoder.feed(encode_pose(pose))[0].sequence, 4)


if __name__ == "__main__":
    unittest.main()
