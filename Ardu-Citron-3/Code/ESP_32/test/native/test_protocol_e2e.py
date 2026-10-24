#!/usr/bin/env python3
"""Software-only vision -> UART-frame contract smoke test."""
import pathlib
import subprocess
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DETECTOR_DIR = ROOT.parents[1] / "Sol"
PI_DIR = ROOT.parents[0] / "Pi"
sys.path.insert(0, str(DETECTOR_DIR))
sys.path.insert(0, str(PI_DIR))
import aruco_dedup_v2 as aruco
from vision_uart import StreamDecoder, VisionPose, encode_pose


def main() -> None:
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 120)
    image = np.full((240, 320), 245, np.uint8)
    image[60:180, 100:220] = marker
    detector = aruco.ArucoCleanDetector(preset="balanced", roi_tracking=False)
    detections, _ = detector.detect(image, t=1.0)
    assert len(detections) == 1 and detections[0].marker_id == 7, detections
    quality = detector.scorer(image, detections[0])
    assert 0.0 <= quality <= 1.0
    x_px, y_px = detections[0].cx, detections[0].cy
    pose = VisionPose(1, 1000, x_px, y_px, 0.0, 0.0, 0.0, 0.0, quality)
    decoder = StreamDecoder()
    decoded_pose = decoder.feed(encode_pose(pose))
    assert len(decoded_pose) == 1 and decoded_pose[0].sequence == 1

    # The native suite validates exact byte encoding/decoding, CRC, sequence
    # rollover and safe mode gating. Pose cannot arm without an RC frame.
    result = subprocess.run([
        sys.executable, str(ROOT / "test/native/run_native.py"),
        "--quality", f"{quality:.6f}",
    ], check=True, capture_output=True, text=True)
    print(result.stdout, end="")
    print(f"E2E vision -> UART codec passed: ArUco id=7, quality={quality:.3f}")


if __name__ == "__main__":
    main()
