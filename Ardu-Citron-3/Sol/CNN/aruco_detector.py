#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class ArucoDetector:
    """
    Détecteur ArUco utilisant le dictionnaire OpenCV DICT_4X4_1000.

    Fonction principale :
        detect(image)

    Retour :
        [
            {
                "id": int,
                "corners": ndarray (4x2),
                "center": (x, y)
            },
            ...
        ]
    """

    def __init__(self):

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_1000
        )

        self.parameters = cv2.aruco.DetectorParameters()

        # Raffinement subpixel
        self.parameters.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX
        )

        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary,
            self.parameters
        )

    def detect(self, image):

        if image is None:
            return []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        corners, ids, rejected = self.detector.detectMarkers(gray)

        markers = []

        if ids is None:
            return markers

        ids = ids.flatten()

        for marker_id, pts in zip(ids, corners):

            pts = pts.reshape((4, 2)).astype(np.float32)

            center = pts.mean(axis=0)

            markers.append(
                {
                    "id": int(marker_id),
                    "corners": pts,
                    "center": (
                        float(center[0]),
                        float(center[1]),
                    ),
                }
            )

        return markers

    def draw(self, image, markers):

        output = image.copy()

        for marker in markers:

            pts = marker["corners"].astype(np.int32)

            cv2.polylines(
                output,
                [pts],
                True,
                (0, 255, 0),
                2,
            )

            cx, cy = map(int, marker["center"])

            cv2.circle(
                output,
                (cx, cy),
                4,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                output,
                str(marker["id"]),
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        return output

    def detect_and_draw(self, image):

        markers = self.detect(image)

        result = self.draw(image, markers)

        return result, markers
