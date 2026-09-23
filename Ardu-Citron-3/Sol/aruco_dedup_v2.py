#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aruco_dedup_v2.py — detection ArUco sans doublons, faible latence,
pour Raspberry Pi Compute Module 4 + Camera Module 3 (Sony IMX708).

Priorites, dans l'ordre : latence > fps reel > fiabilite > qualite visuelle.
Une optimisation qui fait perdre un vrai marqueur est refusee, meme si elle
augmente les fps.

PIPELINE
--------
    libcamera/ISP -> buffer DMA -> plan Y (triple buffer, latest-wins)
        -> planification ROI (predite par les tracks) ou balayage complet
        -> cv2.aruco (jeu de parametres distinct plein cadre / ROI)
        -> pre-filtres geometriques -> groupes de conflit -> arbitrage
        -> tracking temporel (vitesse + confiance)
        -> sortie (affichage / encodage, decouples du thread de detection)

CHOIX STRUCTURANTS
------------------
  * Flux unique YUV420 : le plan Y EST l'image en niveaux de gris attendue
    par ArUco. Aucun debayer RGB, aucun cvtColor par image.
  * Mode capteur binne 1536x864 (~120 fps) ; le 4608x2592 plafonne a ~14 fps
    et n'apporte rien tant que le marqueur fait deja assez de pixels.
  * capture_request + MappedArray + triple buffer : une seule recopie du
    plan Y par image, et la requete libcamera est relachee immediatement
    (jamais d'acces a un buffer apres release()).
  * latest-wins : si la detection prend du retard, les images anciennes
    sont abandonnees plutot que mises en file.
  * Score de qualite paresseux : le warpPerspective (~150 us/marqueur sur
    x86, davantage sur ARM) n'est calcule que pour les candidats reellement
    en conflit.
  * Une seule intersection polygonale par paire : IoU et taux d'imbrication
    sont derives du meme appel a cv2.intersectConvexConvex.

REGLAGES CM4 (hors script)
--------------------------
    /boot/firmware/config.txt :
        camera_auto_detect=0
        dtoverlay=imx708,cam0         # cam1 selon le port CSI de la carte IO
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

DIMENSIONNEMENT (IMX708 : f=4,74 mm, pixels 1,4 um, 4608 px de large)
---------------------------------------------------------------------
    focale en pixels a la largeur W  :  f_px ~= 0,735 * W
    taille du marqueur a l'image     :  px ~= 0,735 * W * S / Z
      S = cote du marqueur (m), Z = distance (m)
    Il faut ~5 px par module pour un decodage fiable, soit ~30 px de cote
    pour un 4x4 borde (6 modules). Exemple : W=768, S=0,10 m, Z=3 m
    -> ~19 px : trop juste. Monter en W ou en S, pas en mode capteur.
    Formule de premier ordre (paraxiale, sans distorsion) : a verifier par
    mesure sur la cible.

USAGE
-----
    python3 aruco_dedup_v2.py --picam --preset balanced --no-display --bench 1000
    python3 aruco_dedup_v2.py --picam --preset speed --profile --report
    python3 aruco_dedup_v2.py --image photo.jpg --preset quality --report
    python3 aruco_dedup_v2.py --video vol.mp4 --out annote.mp4 --stabilize
    python3 aruco_dedup_v2.py --picam --custom-dict mes_marqueurs.npy --unique-ids
"""

from __future__ import annotations

import argparse
import logging
import math
import queue
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as _e:                                    # [ROBUSTESSE]
    raise SystemExit("OpenCV introuvable : sudo apt install -y python3-opencv") from _e

if not hasattr(cv2, "aruco"):                                # [ROBUSTESSE]
    raise SystemExit("OpenCV compile sans le module aruco (paquet opencv-contrib requis)")

log = logging.getLogger("aruco")

__version__ = "2.1"

# CHANGELOG 2.0 -> 2.1 (revue technique + bascule marqueurs 20cm -> 29,7cm) :
#   - Track : vitesse en px/s + dt reel par appel (horodatage), plus une
#     hypothese implicite dt=1 image. Corrige une sous-estimation de la
#     marge des ROI des que l'acquisition latest-wins saute des images.
#   - TemporalStabilizer.update(dets, now) : bug corrige — un track qui
#     vient de naitre n'etait pas marque "apparie" sur sa propre image de
#     naissance et subissait un coast() immediat (hits=1, miss=1 des sa
#     creation). matched_dets (jamais exploite) supprime.
#   - ROI minMarkerPerimeterRate : n'est plus une constante relative au
#     crop (0,03 sur un crop de 170 px ne visait que ~1,3 px de cote —
#     largement sous tout marqueur exploitable). roi_min_marker_side_px
#     fixe desormais un cote minimal ABSOLU ; chaque taille de ROI
#     rencontree obtient son propre ArucoDetector, mis en cache par classe
#     de taille (pas de reconstruction a chaque image).
#   - Recuperation a paliers : ROI -> ROI elargie (roi_retry_margin_scale)
#     -> plein cadre seulement en dernier recours, au lieu d'un passage
#     direct au plein cadre des qu'un track manque a l'appel.
#   - detect() chronometre separement aruco/dedup/tracking (stats
#     t_aruco/t_dedup/t_track) au lieu d'un seul bloc "detect" global.
#   - Preset quality : perim 0,02->0,08 et min_score 0,35->0,42, mesures
#     sur generate_realistic_dataset.py (15->4 faux positifs, rappel
#     inchange) — voir le commentaire sur PRESETS.


# ==========================================================================
# Compatibilite OpenCV
# ==========================================================================
def _set_if_exists(obj, name: str, value) -> bool:
    """Affecte un parametre seulement s'il existe dans cette version d'OpenCV."""
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
            return True
        except Exception as e:                               # pragma: no cover
            log.debug("parametre %s refuse : %s", name, e)
    return False


def _new_detector_params():
    a = cv2.aruco
    if hasattr(a, "DetectorParameters"):
        try:
            return a.DetectorParameters()
        except TypeError:                                    # OpenCV 4.6 et avant
            pass
    return a.DetectorParameters_create()


def _make_detect_fn(dictionary, params) -> Callable:
    """Retourne detect(img) -> (corners, ids, rejected), API 4.7+ ou ancienne."""
    a = cv2.aruco
    if hasattr(a, "ArucoDetector"):
        det = a.ArucoDetector(dictionary, params)
        return det.detectMarkers
    return lambda img: a.detectMarkers(img, dictionary, parameters=params)


# ==========================================================================
# Detection
# ==========================================================================
class Detection:
    """Un candidat marqueur.

    Classe a __slots__ plutot que dataclass : elle est instanciee a chaque
    marqueur de chaque image, et la geometrie est calculee en arithmetique
    scalaire Python (shoelace, min/max) plutot qu'en appels numpy/OpenCV.
    Mesure x86 : 11,0 us -> 2,2 us par instance.
    """

    __slots__ = ("marker_id", "corners", "score", "area", "cx", "cy",
                 "x0", "y0", "x1", "y1", "scale")

    def __init__(self, marker_id: int, corners: np.ndarray,
                 score: Optional[float] = None):
        if corners.dtype != np.float32 or corners.shape != (4, 2) \
                or not corners.flags["C_CONTIGUOUS"]:
            corners = np.ascontiguousarray(corners, dtype=np.float32).reshape(4, 2)
        self.marker_id = int(marker_id)
        self.corners = corners
        self.score = score

        (ax, ay), (bx, by), (cx_, cy_), (dx, dy) = corners.tolist()
        self.area = 0.5 * abs(ax * by - bx * ay + bx * cy_ - cx_ * by
                              + cx_ * dy - dx * cy_ + dx * ay - ax * dy)
        self.cx = 0.25 * (ax + bx + cx_ + dx)
        self.cy = 0.25 * (ay + by + cy_ + dy)
        self.x0 = ax if ax < bx else bx
        if cx_ < self.x0:
            self.x0 = cx_
        if dx < self.x0:
            self.x0 = dx
        self.x1 = ax if ax > bx else bx
        if cx_ > self.x1:
            self.x1 = cx_
        if dx > self.x1:
            self.x1 = dx
        self.y0 = ay if ay < by else by
        if cy_ < self.y0:
            self.y0 = cy_
        if dy < self.y0:
            self.y0 = dy
        self.y1 = ay if ay > by else by
        if cy_ > self.y1:
            self.y1 = cy_
        if dy > self.y1:
            self.y1 = dy
        self.scale = math.sqrt(self.area) if self.area > 0.0 else 0.0

    @property
    def center(self) -> np.ndarray:
        return np.array((self.cx, self.cy), dtype=np.float32)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def __repr__(self):
        return f"Detection(id={self.marker_id}, area={self.area:.0f}, score={self.score})"


# ==========================================================================
# Geometrie
# ==========================================================================
def poly_area(pts) -> float:
    """Aire d'un polygone convexe (shoelace ; 0,43 us contre 0,57 us pour
    cv2.contourArea, et surtout sans conversion d'entree)."""
    p = pts.tolist() if isinstance(pts, np.ndarray) else list(pts)
    n = len(p)
    s = 0.0
    for i in range(n):
        x0, y0 = p[i]
        x1, y1 = p[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * abs(s)


def bbox_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    """Pre-filtre O(1) en arithmetique scalaire."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def intersect_area(a: np.ndarray, b: np.ndarray) -> float:
    """Aire d'intersection de deux quads convexes (gere l'imbrication)."""
    try:
        area, _ = cv2.intersectConvexConvex(a, b)
    except cv2.error:                                        # quad degenere
        return 0.0
    return float(area)


def overlap_metrics(da: "Detection", db: "Detection") -> Tuple[float, float]:
    """(IoU, taux d'imbrication) en UN SEUL appel a intersectConvexConvex.

    La v1 appelait iou() puis containment(), donc deux intersections
    polygonales pour la meme paire : 60,7 us contre 30,5 us ici (x86).
    """
    inter = intersect_area(da.corners, db.corners)
    if inter <= 0.0:
        return 0.0, 0.0
    union = da.area + db.area - inter
    small = da.area if da.area < db.area else db.area
    return (inter / union if union > 1e-6 else 0.0,
            inter / small if small > 1e-6 else 0.0)


def iou(a, b) -> float:
    """Conservee pour compatibilite ; prefere overlap_metrics() en interne."""
    aa, bb = (a if isinstance(a, Detection) else Detection(0, a),
              b if isinstance(b, Detection) else Detection(0, b))
    return overlap_metrics(aa, bb)[0]


def containment(a, b) -> float:
    """Conservee pour compatibilite ; prefere overlap_metrics() en interne."""
    aa, bb = (a if isinstance(a, Detection) else Detection(0, a),
              b if isinstance(b, Detection) else Detection(0, b))
    return overlap_metrics(aa, bb)[1]


def quad_sanity(d: "Detection") -> float:
    """Coherence geometrique dans [0, 1], SANS warp (~2 us).

    Rapport du plus petit au plus grand cote et rapport des diagonales : un
    quad plausible sous perspective garde des cotes du meme ordre et des
    diagonales comparables. Sert a departager deux candidats avant de payer
    un warpPerspective, et a penaliser les quads aberrants.
    """
    p = d.corners.tolist()
    sides = []
    for i in range(4):
        x0, y0 = p[i]
        x1, y1 = p[(i + 1) % 4]
        sides.append(math.hypot(x1 - x0, y1 - y0))
    smin, smax = min(sides), max(sides)
    if smax <= 1e-6:
        return 0.0
    side_ratio = smin / smax
    d1 = math.hypot(p[2][0] - p[0][0], p[2][1] - p[0][1])
    d2 = math.hypot(p[3][0] - p[1][0], p[3][1] - p[1][1])
    diag_ratio = min(d1, d2) / max(d1, d2) if max(d1, d2) > 1e-6 else 0.0
    # Moyenne geometrique et non arithmetique : un quad tres aplati garde des
    # diagonales quasi egales, seul le rapport des cotes s'effondre. Une
    # somme ponderee le noterait a 0,5, le produit le sanctionne. Un vrai
    # marqueur vu sous un angle rasant reste vers 0,5-0,6.
    return float(math.sqrt(side_ratio * diag_ratio))


# ==========================================================================
# Score de qualite
# ==========================================================================
_CELL_PX = 6            # 6 px par module -> 36x36 pour un 4x4 borde
_MARGIN = 1             # 1 px de bord de module ignore (flou d'interpolation)


class QualityScorer:
    """Score de qualite d'un marqueur, calcule sur le patch redresse.

    Le buffer de destination et la matrice de coins cible sont prealloues :
    l'appel ne fait plus d'allocation numpy en dehors de warpPerspective.
    """

    __slots__ = ("marker_size", "border_bits", "n", "side", "_dst", "_inner_mask")

    def __init__(self, marker_size: int = 4, border_bits: int = 1):
        self.marker_size = int(marker_size)
        self.border_bits = int(border_bits)
        self.n = self.marker_size + 2 * self.border_bits
        self.side = self.n * _CELL_PX
        s = float(self.side)
        self._dst = np.array([[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]], dtype=np.float32)
        m = np.zeros((self.n, self.n), dtype=bool)
        m[self.border_bits:self.n - self.border_bits,
          self.border_bits:self.n - self.border_bits] = True
        self._inner_mask = m

    def __call__(self, gray: np.ndarray, d: "Detection") -> float:
        geom = quad_sanity(d)
        if geom < 0.15:                 # quad aberrant : inutile de warper
            return 0.05 * geom

        try:
            H = cv2.getPerspectiveTransform(d.corners, self._dst)
            patch = cv2.warpPerspective(gray, H, (self.side, self.side),
                                        flags=cv2.INTER_LINEAR)
        except cv2.error:
            return 0.0
        if patch.size == 0:
            return 0.0

        thr, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Moyenne du coeur de chaque module, sans boucle Python.
        cells = patch.reshape(self.n, _CELL_PX, self.n, _CELL_PX)
        core = cells[:, _MARGIN:_CELL_PX - _MARGIN, :, _MARGIN:_CELL_PX - _MARGIN]
        means = core.mean(axis=(1, 3), dtype=np.float32)

        border_cells = means[~self._inner_mask]
        border_purity = float(np.count_nonzero(border_cells < thr)) / border_cells.size

        inner = means[self._inner_mask]
        spread = float(inner.max() - inner.min())
        if spread < 1.0:
            spread = 1.0
        margin = float(np.abs(inner - thr).mean() / (0.5 * spread))
        if margin > 1.0:
            margin = 1.0

        dark = patch <= thr
        nd = int(np.count_nonzero(dark))
        if 0 < nd < patch.size:
            contrast = float((patch[~dark].mean() - patch[dark].mean()) / 255.0)
        else:
            contrast = 0.0

        q = 0.42 * border_purity + 0.26 * margin + 0.17 * contrast + 0.15 * geom
        return float(q if q < 1.0 else 1.0)


def marker_quality(gray: np.ndarray, corners, marker_size: int = 4,
                   border_bits: int = 1) -> float:
    """Interface v1 conservee (alloue un scorer a chaque appel : pour les
    tests et les usages hors boucle chaude)."""
    d = corners if isinstance(corners, Detection) else Detection(0, corners)
    return QualityScorer(marker_size, border_bits)(gray, d)


# ==========================================================================
# Deduplication
# ==========================================================================
class _Union:
    __slots__ = ("p",)

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        p = self.p
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def join(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def deduplicate(dets: List["Detection"],
                score_fn: Optional[Callable[["Detection"], float]] = None,
                iou_thresh: float = 0.35,
                nested_thresh: float = 0.75,
                center_ratio: float = 0.5,
                cross_id_iou: float = 0.50,
                cross_id_nested: float = 0.85,
                nested_area_min: float = 0.08,
                unique_ids: bool = False,
                min_score: float = 0.0,
                stats: Optional[Dict[str, int]] = None
                ) -> Tuple[List["Detection"], List[Tuple["Detection", str]]]:
    """Regroupe les candidats en conflit puis garde le meilleur de chaque groupe.

    Chaine de pre-filtres avant toute intersection polygonale (30 us) :
        1. bbox disjointes                      -> aucun conflit possible
        2. centres eloignes de > 0,75*max(cote) -> ni recouvrement ni imbrication
        3. rapport d'aires < nested_area_min    -> petit marqueur reellement
           distinct a l'interieur d'un grand, pas un cadre parasite
    Les paires sont parcourues par balayage sur x0 trie : des que
    dets[j].x0 > dets[i].x1, tous les suivants sont disjoints, on sort.
    Complexite O(n log n + k) avec k le nombre de paires qui se recouvrent
    vraiment, au lieu de O(n^2). Aucune structure spatiale n'est construite :
    en usage reel n reste petit (< 30) et une grille couterait plus cher que
    le balayage.

    Le score n'est evalue que pour les groupes de taille > 1 (et pour tous
    si min_score > 0) : sans doublon, le correctif ne coute rien.

    Retourne (gardes, [(supprime, motif), ...]).
    """
    n = len(dets)
    if n == 0:
        return [], []
    if n == 1 and not (min_score > 0.0):                     # chemin rapide
        return [dets[0]], []

    def sc(d: "Detection") -> float:
        if d.score is None:
            d.score = score_fn(d) if score_fn is not None else 0.0
            if stats is not None:
                stats["scored"] = stats.get("scored", 0) + 1
        return d.score

    uf = _Union(n)
    reasons: Dict[frozenset, str] = {}
    n_pairs = 0

    order = sorted(range(n), key=lambda i: dets[i].x0)
    for oi in range(n):
        i = order[oi]
        di = dets[i]
        for oj in range(oi + 1, n):
            j = order[oj]
            dj = dets[j]
            if dj.x0 > di.x1:                                # balayage : fin
                break
            if dj.y1 < di.y0 or di.y1 < dj.y0:               # 1. bbox en y
                continue

            same_id = (di.marker_id == dj.marker_id)
            dist = math.hypot(di.cx - dj.cx, di.cy - dj.cy)
            ref = di.scale if di.scale > dj.scale else dj.scale
            if dist > 0.75 * ref:                            # 2. centres loin
                continue

            n_pairs += 1
            ov_iou, ov_nest = overlap_metrics(di, dj)
            amin, amax = ((di.area, dj.area) if di.area < dj.area
                          else (dj.area, di.area))
            area_ratio = amin / amax if amax > 1e-6 else 0.0
            nested = (ov_nest > nested_thresh and area_ratio > nested_area_min)

            if same_id:
                mean_scale = 0.5 * (di.scale + dj.scale)
                if ov_iou > iou_thresh:
                    r = f"meme ID, IoU={ov_iou:.2f}"
                elif nested:
                    r = f"quads imbriques (inclusion {ov_nest:.2f})"
                elif dist < center_ratio * mean_scale:
                    r = f"meme ID, centres a {dist:.1f}px"
                else:
                    continue
            else:
                # Deux marqueurs differents ne se recouvrent jamais autant :
                # c'est un decodage parasite sur un quad deja explique.
                if ov_iou > cross_id_iou or (ov_nest > cross_id_nested
                                             and area_ratio > nested_area_min):
                    r = f"conflit ID {di.marker_id}/{dj.marker_id}"
                else:
                    continue

            uf.join(i, j)
            reasons[frozenset((i, j))] = r

    if unique_ids:
        # Arbitrage global par ID : O(n), sans test geometrique.
        by_id: Dict[int, int] = {}
        for i, d in enumerate(dets):
            first = by_id.setdefault(d.marker_id, i)
            if first != i:
                uf.join(first, i)
                reasons.setdefault(frozenset((first, i)), "id-unique")

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    kept: List["Detection"] = []
    removed: List[Tuple["Detection", str]] = []

    for members in groups.values():
        if len(members) == 1:
            d = dets[members[0]]
            if min_score > 0.0 and sc(d) < min_score:
                removed.append((d, f"score faible {d.score:.2f}"))
            else:
                kept.append(d)
            continue
        # Arbitrage : qualite d'abord ; a qualite egale le plus petit quad,
        # car dans une imbrication le vrai marqueur est la bordure interieure
        # (le cadre exterieur decode par accident sur le meme ID). La regle
        # n'est appliquee qu'a egalite de score : elle ne prime jamais sur la
        # mesure de qualite.
        best = max(members, key=lambda k: (round(sc(dets[k]), 3), -dets[k].area))
        d = dets[best]
        if min_score > 0.0 and sc(d) < min_score:
            removed.append((d, f"score faible {d.score:.2f}"))
        else:
            kept.append(d)
        for k in members:
            if k == best:
                continue
            why = reasons.get(frozenset((k, best)))
            if why is None:
                why = next((v for key, v in reasons.items() if k in key), "doublon")
            removed.append((dets[k], why))

    if stats is not None:
        stats["pairs"] = stats.get("pairs", 0) + n_pairs

    kept.sort(key=lambda x: x.marker_id)
    return kept, removed


# ==========================================================================
# Tracking temporel
# ==========================================================================
class Track:
    """Etat d'un marqueur suivi : position, vitesse (px/s), confiance.

    Vitesse en px/s et non px/frame : l'acquisition est latest-wins (voir
    PiCam3Source), donc le nombre d'images realites entre deux appels a
    detect() varie — 1 en charge normale, plusieurs si la detection prend
    du retard. Raisonner en "px/frame" revient a supposer dt constant, ce
    qui sous-estime la vitesse reelle (et donc la marge des ROI) des que
    des images sont sautees. predict()/update()/coast() prennent donc un
    dt explicite, jamais implicite.
    """

    __slots__ = ("marker_id", "corners", "cx", "cy", "vx", "vy", "scale",
                 "score", "hits", "miss", "age", "last_t")

    def __init__(self, d: "Detection", t: float):
        self.marker_id = d.marker_id
        self.corners = d.corners.copy()
        self.cx, self.cy = d.cx, d.cy
        self.vx = self.vy = 0.0
        self.scale = d.scale
        self.score = d.score if d.score is not None else 0.5
        self.hits = 1
        self.miss = 0
        self.age = 1
        self.last_t = t

    @property
    def confidence(self) -> float:
        """[0, 1] : monte avec les detections consecutives, chute avec les
        images manquees. Pilote la marge des ROI et la frequence des
        balayages complets."""
        h = self.hits if self.hits < 8 else 8
        return max(0.0, (h / 8.0) * (1.0 - 0.3 * self.miss))

    def predict(self, dt: float) -> Tuple[float, float]:
        """Position estimee dt SECONDES apres la derniere observation."""
        return self.cx + self.vx * dt, self.cy + self.vy * dt

    def update(self, d: "Detection", alpha: float, t: float) -> None:
        dt = max(t - self.last_t, 1e-3)          # plancher : evite une div/0
        pcx, pcy = d.cx, d.cy                     # ou un pic de vitesse si deux
        nvx = (pcx - self.cx) / dt                # detections arrivent au meme t
        nvy = (pcy - self.cy) / dt
        self.vx = 0.6 * self.vx + 0.4 * nvx
        self.vy = 0.6 * self.vy + 0.4 * nvy
        self.cx, self.cy = pcx, pcy
        np.multiply(self.corners, 1.0 - alpha, out=self.corners)
        self.corners += alpha * d.corners
        self.scale = alpha * d.scale + (1.0 - alpha) * self.scale
        if d.score is not None:
            self.score = alpha * d.score + (1.0 - alpha) * self.score
        self.hits += 1
        self.miss = 0
        self.age += 1
        self.last_t = t

    def coast(self, t: float) -> None:
        """Image sans detection : on extrapole la position sur le dt REEL
        ecoule (pas 1 frame) pour garder une ROI utile, sans jamais publier
        le marqueur (c'est la persistance qui decide de la publication)."""
        dt = t - self.last_t
        self.cx += self.vx * dt
        self.cy += self.vy * dt
        self.miss += 1
        self.age += 1
        self.last_t = t


class TemporalStabilizer:
    """Association + lissage + persistance.

    Association a deux niveaux, du moins cher au plus cher :
      1. meme ID et distance a la position PREDITE < gate * cote ;
      2. a egalite, la plus petite variation d'aire departage.
    Pas de Hongrois ni de filtre de Kalman : sur CM4, avec moins d'une
    dizaine de marqueurs, le cout de l'association n'est pas le probleme,
    et un Kalman a 8 etats par marqueur coute plus qu'il ne rapporte.
    """

    def __init__(self, alpha: float = 0.5, min_hits: int = 2, max_miss: int = 3,
                 gate: float = 0.75, publish_coasted: bool = False):
        self.alpha = alpha
        self.min_hits = min_hits
        self.max_miss = max_miss
        self.gate = gate
        self.publish_coasted = publish_coasted
        self.tracks: List[Track] = []

    def update(self, dets: List["Detection"], now: float) -> List["Detection"]:
        """now : horodatage de CETTE image, en secondes, horloge monotone
        (time.perf_counter() par defaut - voir ArucoCleanDetector.detect).
        Sert a calculer un dt reel par track, jamais suppose egal a 1 image."""
        matched_tracks = set()

        for d in dets:
            best, best_cost = None, float("inf")
            for tr in self.tracks:
                if id(tr) in matched_tracks or tr.marker_id != d.marker_id:
                    continue
                px, py = tr.predict(now - tr.last_t)
                dist = math.hypot(px - d.cx, py - d.cy)
                gate = self.gate * max(d.scale, tr.scale)
                if dist >= gate:
                    continue
                da = abs(d.scale - tr.scale) / max(tr.scale, 1e-6)
                cost = dist / max(gate, 1e-6) + 0.5 * da
                if cost < best_cost:
                    best, best_cost = tr, cost
            if best is None:
                # BUG CORRIGE : la version precedente ajoutait le nouveau
                # track SANS l'inscrire dans matched_tracks, si bien que la
                # boucle coast() ci-dessous le "manquait" des sa premiere
                # image (hits=1, miss=1 des la naissance). Desormais un
                # track fraichement cree compte comme apparie sur CETTE image.
                new_tr = Track(d, now)
                self.tracks.append(new_tr)
                matched_tracks.add(id(new_tr))
            else:
                best.update(d, self.alpha, now)
                matched_tracks.add(id(best))

        for tr in self.tracks:
            if id(tr) not in matched_tracks:
                tr.coast(now)
        self.tracks = [tr for tr in self.tracks if tr.miss <= self.max_miss]

        out: List["Detection"] = []
        for tr in self.tracks:
            if tr.hits < self.min_hits:
                continue
            if tr.miss > 0 and not self.publish_coasted:
                continue
            out.append(Detection(tr.marker_id, tr.corners.copy(), tr.score))
        return out


# ==========================================================================
# Planification ROI et politique de balayage
# ==========================================================================
class RoiPlanner:
    """Fenetres de recherche construites sur la position PREDITE des tracks.

    Marge dynamique : un marqueur stable et lent obtient une petite fenetre,
    un marqueur rapide ou peu fiable une grande. La marge couvre au minimum
    le deplacement d'une image, pour qu'un marqueur ne sorte jamais de sa
    propre fenetre entre deux images.
    """

    __slots__ = ("margin", "max_coverage", "min_side")

    def __init__(self, margin: float = 0.45, max_coverage: float = 0.35,
                 min_side: int = 24):
        self.margin = margin
        self.max_coverage = max_coverage
        self.min_side = min_side

    def plan(self, tracks: List[Track], w: int, h: int, expected_dt: float,
             margin_scale: float = 1.0) -> List[Tuple[int, int, int, int]]:
        """expected_dt : dt (s) estime jusqu'au PROCHAIN appel a detect() —
        remplace l'ancienne hypothese implicite "vitesse en px/frame".
        margin_scale > 1 : fenetre volontairement elargie (voir la
        recuperation a paliers dans ArucoCleanDetector.detect)."""
        boxes: List[List[int]] = []
        for tr in tracks:
            px, py = tr.predict(expected_dt)
            speed = math.hypot(tr.vx, tr.vy)          # px/s
            conf = tr.confidence
            # marge = base * (1 + incertitude) + 2 pas de temps de deplacement
            m = margin_scale * (tr.scale * (self.margin * (1.6 - conf))
                                + 2.0 * speed * expected_dt)
            half = 0.5 * tr.scale + m
            x0 = int(px - half); y0 = int(py - half)
            x1 = int(px + half); y1 = int(py + half)
            x0 = 0 if x0 < 0 else x0
            y0 = 0 if y0 < 0 else y0
            x1 = w if x1 > w else x1
            y1 = h if y1 > h else y1
            if x1 - x0 >= self.min_side and y1 - y0 >= self.min_side:
                boxes.append([x0, y0, x1, y1])

        if not boxes:
            return []

        merged: List[List[int]] = []
        for b in sorted(boxes, key=lambda z: z[0]):
            hit = None
            for m in merged:
                if not (b[0] > m[2] or b[2] < m[0] or b[1] > m[3] or b[3] < m[1]):
                    hit = m
                    break
            if hit is None:
                merged.append(b)
            else:
                hit[0] = min(hit[0], b[0]); hit[1] = min(hit[1], b[1])
                hit[2] = max(hit[2], b[2]); hit[3] = max(hit[3], b[3])

        cover = sum((m[2] - m[0]) * (m[3] - m[1]) for m in merged)
        if cover > self.max_coverage * w * h:
            # Decouper coute alors plus cher qu'un balayage direct : les
            # crops impliquent une copie contigue et autant d'appels
            # detectMarkers que de fenetres.
            return []
        return [(m[0], m[1], m[2], m[3]) for m in merged]


class ScanPolicy:
    """Decide balayage complet ou ROI, par confiance plutot que par modulo.

    Balayage complet force si :
      * aucun track ;
      * un track a ete perdu a l'image precedente ;
      * confiance minimale des tracks trop basse ;
      * changement de scene detecte ;
      * intervalle courant atteint.
    L'intervalle part de `base` et double a chaque serie ROI reussie,
    jusqu'a `max_interval` : une scene stable paie de moins en moins de
    balayages complets, une scene agitee revient tout de suite au complet.
    """

    __slots__ = ("base", "max_interval", "scene_thresh", "interval",
                 "since_full", "_sig", "force")

    def __init__(self, base: int = 4, max_interval: int = 32,
                 scene_thresh: float = 12.0):
        self.base = max(1, base)
        self.max_interval = max(self.base, max_interval)
        self.scene_thresh = scene_thresh
        self.interval = self.base
        self.since_full = 10 ** 9
        self._sig: Optional[np.ndarray] = None
        self.force = True

    def scene_changed(self, gray: np.ndarray) -> bool:
        """Signature par sous-echantillonnage par pas de 16 : une VUE numpy,
        aucune copie de l'image, ~1,3 us a 640x480."""
        sig = gray[::16, ::16].astype(np.int16)
        prev, self._sig = self._sig, sig
        if prev is None or prev.shape != sig.shape:
            return True
        return float(np.abs(sig - prev).mean()) > self.scene_thresh

    def want_full(self, tracks: List[Track], gray: np.ndarray) -> Tuple[bool, str]:
        if self.force:
            self.force = False
            return True, "force"
        if not tracks:
            return True, "aucun track"
        if self.scene_changed(gray):
            return True, "changement de scene"
        if min(t.confidence for t in tracks) < 0.35:
            return True, "confiance faible"
        if self.since_full >= self.interval:
            return True, "intervalle"
        return False, ""

    def after_full(self, n_found: int, n_tracks_before: int) -> None:
        self.since_full = 0
        if n_found != n_tracks_before:
            self.interval = self.base          # la scene bouge : on reste dense
        else:
            self.interval = min(self.max_interval, max(self.base, self.interval * 2))

    def after_roi(self, ok: bool) -> None:
        self.since_full += 1
        if not ok:
            self.interval = self.base
            self.force = True                  # rebalayage complet immediat


# ==========================================================================
# Detecteur
# ==========================================================================
#  Presets. Chaque parametre et sa justification :
#
#  adaptiveThreshWinSizeMin/Max/Step
#      Nombre de passes de seuillage adaptatif ; chaque passe refait un
#      seuillage plein cadre + une extraction de contours. C'est le poste le
#      plus lourd de detectMarkers. Plus de passes = meilleur rappel quand
#      l'eclairage est heterogene ou les marqueurs de tailles tres
#      differentes ; moins de passes = gain direct et important.
#  minMarkerPerimeterRate
#      Perimetre minimal, en fraction de la plus grande dimension de
#      l'image. Le relever ecarte tot une majorite de contours parasites
#      (donc gros gain) au prix des marqueurs les plus lointains : c'est le
#      parametre qui arbitre vitesse contre portee.
#  maxMarkerPerimeterRate
#      Borne haute ; 4,0 laisse passer un marqueur qui remplit l'image.
#  polygonalApproxAccuracyRate
#      Tolerance de l'approximation polygonale. Trop haut : des contours non
#      carres passent (faux positifs) ; trop bas : les marqueurs flous ou
#      tres inclines sont rejetes (faux negatifs).
#  perspectiveRemovePixelPerCell
#      Pixels par module lors du redressement pour lire les bits. 3 a 4
#      suffisent pour un 4x4 ; au-dela on paie un warp plus gros par
#      candidat sans gagner en fiabilite de decodage.
#  perspectiveRemoveIgnoredMarginPerCell
#      Fraction du module ignoree sur les bords ; protege du flou de bord.
#  cornerRefinementMethod / WinSize / MaxIterations
#      SUBPIX donne des coins au ~1/10 px : indispensable pour une pose
#      (solvePnP), inutile pour compter ou identifier. NONE laisse ~0,5 px
#      d'erreur et supprime un cornerSubPix par marqueur.
#  useAruco3Detection + minSideLengthCanonicalImg
#      Detection sur image reduite puis raffinement a pleine resolution.
#      Gros gain plein cadre ; en revanche le sous-echantillonnage peut
#      faire passer un petit marqueur sous minSideLengthCanonicalImg et le
#      faire disparaitre : d'ou un jeu de parametres distinct pour les ROI.
#  errorCorrectionRate
#      Fraction des bits corrigeables reellement utilisee. Monter = moins de
#      faux negatifs sur marqueur abime, mais plus de faux positifs,
#      d'autant plus que la distance minimale du dictionnaire est faible.
#
PRESETS: Dict[str, Dict] = {
    # perim=0.08 et min_score=0.42 (au lieu de 0.02 / 0.35) : mesure sur un
    # dataset de scenes realistes (sol textures, flou de mouvement, bruit
    # capteur, JPEG — voir generate_realistic_dataset.py) avec le
    # dictionnaire DICT_4X4_1000. Sans ce resserrement, 15 detections sur
    # 61 candidats "gardes" etaient des faux positifs decodes par hasard
    # sur du bruit de texture (aire 7-163 px^2, sous 13 px de cote) ou des
    # marqueurs authentiques trop degrades pour dater fiablement leur ID.
    # perim=0.08 (min ~13 px de cote a 640 px de large) ecarte les
    # candidats microscopiques AVANT tout decodage : mesure, aucun
    # marqueur reellement detecte dans ce dataset ne fait moins de 14 px
    # de cote, donc aucun cout de rappel. min_score=0.42 acheve d'ecarter
    # les faux positifs de plus grande taille mais de mauvaise qualite de
    # patch. Ensemble : 15 -> 4 faux positifs, rappel inchange. Les 4
    # restants sont de VRAIS marqueurs decodes SANS aucune erreur vers un
    # MAUVAIS ID valide du dictionnaire (confirme : faire varier
    # errorCorrectionRate de 0,6 a 0,15 n'en change aucun) — un probleme
    # de collision du dictionnaire 1000 codes, pas de reglage du
    # detecteur ; voir la note plus bas sur le choix de dictionnaire.
    "quality": dict(win=(3, 23, 10), aruco3=False, refine="subpix", perim=0.08,
                    ppc=5, poly=0.05, ecr=0.6, min_score=0.42),
    "balanced": dict(win=(5, 15, 10), aruco3=True, refine="subpix", perim=0.04,
                     ppc=4, poly=0.05, ecr=0.6, min_score=0.0),
    "speed": dict(win=(7, 7, 1), aruco3=True, refine="none", perim=0.06,
                  ppc=3, poly=0.06, ecr=0.5, min_score=0.0),
}
PRESETS["fast"] = PRESETS["speed"]          # alias

# CHOIX DU DICTIONNAIRE : a taille de marqueur egale, DICT_4X4_1000 (1000
# mots de code dans un espace de 16 bits) a une distance de Hamming
# minimale bien plus faible que DICT_4X4_50. Mesure sur le meme pipeline
# de scenes realistes, meme nombre de marqueurs place : DICT_4X4_1000 a
# produit 6 faux positifs (dont 4 marqueurs reels decodes sans erreur vers
# le mauvais ID) ; DICT_4X4_50, dans les memes conditions, 0 — sur aucun
# des deux plans (faux positifs sur du bruit, mesdecodage d'un marqueur
# reel). Aucun reglage du detecteur ne corrige ce que le dictionnaire
# choisi rend intrinsequement ambigu : utiliser le plus PETIT dictionnaire
# predefini qui couvre le nombre de marqueurs reellement deployes.


class ArucoCleanDetector:
    """Detection + deduplication + tracking, plein cadre ou par ROI."""

    def __init__(self,
                 dict_name: str = "DICT_4X4_50",
                 custom_dict: Optional[str] = None,
                 preset: str = "balanced",
                 unique_ids: bool = False,
                 iou_thresh: float = 0.35,
                 nested_thresh: float = 0.75,
                 min_score: Optional[float] = None,
                 stabilize: bool = False,
                 roi_tracking: bool = True,
                 full_scan_every: int = 32,
                 roi_margin: float = 0.45,
                 roi_max_coverage: float = 0.35,
                 roi_min_marker_side_px: float = 12.0,
                 roi_retry_margin_scale: float = 1.7,
                 track_max_miss: int = 3,
                 max_correction: Optional[int] = None):
        self.unique_ids = unique_ids
        self.iou_thresh = iou_thresh
        self.nested_thresh = nested_thresh
        self.roi_tracking = roi_tracking
        self.roi_retry_margin_scale = roi_retry_margin_scale

        cfg = PRESETS.get(preset)
        if cfg is None:
            raise ValueError(f"preset inconnu : {preset} (choix : {list(PRESETS)})")
        self.preset = preset
        self._preset_cfg = cfg
        self.min_score = cfg["min_score"] if min_score is None else float(min_score)

        self.dictionary, self.marker_size = self._load_dictionary(
            dict_name, custom_dict, max_correction)
        self.scorer = QualityScorer(self.marker_size, 1)

        self.params_full = self._build_params(cfg, roi=False)
        self._detect_full = _make_detect_fn(self.dictionary, self.params_full)

        # minMarkerPerimeterRate des ROI : PAS une constante fixe (BUG
        # CORRIGE — voir plus bas). OpenCV l'interprete relativement a la
        # PLUS GRANDE DIMENSION DE L'IMAGE FOURNIE, donc une meme valeur sur
        # un crop de 170 px et sur un plein cadre de 768 px ne vise pas du
        # tout le meme perimetre absolu : perim=0.03 sur un crop de 170 px
        # n'exige que ~5 px de perimetre (donc ~1,3 px de cote), largement
        # sous tout marqueur exploitable, ce qui envoie inutilement des
        # candidats microscopiques dans les etapes couteuses (rectification,
        # decodage). roi_min_marker_side_px fixe a la place un COTE MINIMAL
        # ABSOLU (12 px par defaut) ; chaque taille de ROI rencontree
        # calcule son propre minMarkerPerimeterRate = 4*cote_min/max(w,h) et
        # construit un ArucoDetector DEDIE, mis en cache par taille de ROI
        # arrondie (pas a 32 px) pour ne jamais reconstruire un detecteur a
        # chaque image — seulement a chaque NOUVELLE classe de taille de ROI
        # rencontree (typiquement 1 a 3 par session, le cache est de toute
        # facon borne a 32 entrees par securite).
        self.roi_min_marker_side_px = roi_min_marker_side_px
        self._roi_cache: Dict[Tuple[int, int], Callable] = {}

        # Le tracker sert TOUJOURS (prediction des ROI) ; --stabilize ne
        # decide que de la publication lissee.
        self.tracker = TemporalStabilizer(max_miss=track_max_miss)
        self.stabilize = stabilize
        self.planner = RoiPlanner(margin=roi_margin, max_coverage=roi_max_coverage)
        self.policy = ScanPolicy(base=4, max_interval=max(4, full_scan_every))

        # Horodatage reel entre appels a detect() : voir Track/dt plus haut.
        # L'estimateur d'intervalle (EMA) demarre a 1/30 s, une hypothese
        # median raisonnable, corrigee des le deuxieme appel.
        self._last_call_t: Optional[float] = None
        self._ema_dt: float = 1.0 / 30.0

        self.stats: Dict[str, float] = {}
        self.last_mode = "full"
        self.last_rois = 0
        self.last_raw = 0

    # ---- construction ----------------------------------------------------
    @staticmethod
    def _load_dictionary(dict_name, custom_dict, max_correction):
        a = cv2.aruco
        if custom_dict:
            bits = np.load(custom_dict)
            if bits.ndim != 3 or bits.shape[1] != bits.shape[2]:
                raise ValueError(
                    f"dictionnaire maison : forme (N, k, k) attendue, recue {bits.shape}")
            bits = bits.astype(np.uint8)
            if not hasattr(a, "Dictionary") or not hasattr(a.Dictionary,
                                                           "getByteListFromBits"):
                raise RuntimeError(
                    "cette version d'OpenCV n'expose pas Dictionary.getByteListFromBits ; "
                    "OpenCV >= 4.7 requis pour --custom-dict")
            byte_list = np.concatenate(
                [a.Dictionary.getByteListFromBits(b) for b in bits], axis=0)
            k = int(bits.shape[1])
            dictionary = a.Dictionary(byte_list, k)
            # maxCorrectionBits : nombre de bits corrigeables. Pour une
            # distance minimale dmin, la borne theorique est (dmin-1)//2.
            # Au-dela, on corrige vers un mot de code voisin : les faux
            # positifs explosent. En dessous, un marqueur legerement abime
            # est rejete (faux negatif). Par defaut on ne touche a rien.
            if max_correction is not None:
                dictionary.maxCorrectionBits = int(max_correction)
                log.info("maxCorrectionBits force a %d", max_correction)
            return dictionary, k

        if not hasattr(a, dict_name):
            raise ValueError(f"dictionnaire inconnu : {dict_name}")
        get = (a.getPredefinedDictionary if hasattr(a, "getPredefinedDictionary")
               else a.Dictionary_get)
        dictionary = get(getattr(a, dict_name))
        if max_correction is not None:
            _set_if_exists(dictionary, "maxCorrectionBits", int(max_correction))
        return dictionary, int(getattr(dictionary, "markerSize", 4))

    def _build_params(self, cfg: Dict, roi: bool, perim_override: Optional[float] = None):
        a = cv2.aruco
        p = _new_detector_params()
        # Le chemin ROI a son propre reglage, INDEPENDANT du preset :
        #  * il opere sur quelques milliers de pixels, il n'est donc jamais
        #    le poste dominant -> on y privilegie le rappel (2 passes de
        #    seuillage, raffinement sous-pixel) ;
        #  * ArUco3 sous-echantillonne l'image : sur un petit crop le
        #    marqueur passe sous minSideLengthCanonicalImg et disparait,
        #    donc desactive ;
        #  * polygonalApproxAccuracyRate est resserre : sur un crop bruite,
        #    une tolerance lache transforme le bruit en quadrilateres a
        #    decoder. Mesure x86 sur un crop de 169x169 : 2,40 ms avec la
        #    valeur du preset speed (0,06) contre 1,54 ms a 0,045.
        #  * minMarkerPerimeterRate : perim_override, calcule PAR TAILLE DE
        #    ROI par _get_roi_detect_fn (voir sa docstring) — jamais une
        #    constante relative au crop.
        if roi:
            wmin, wmax, wstep = 5, 15, 10
            perim = perim_override if perim_override is not None else 0.09
            poly, ppc = 0.045, 4
        else:
            wmin, wmax, wstep = cfg["win"]
            perim, poly, ppc = cfg["perim"], cfg["poly"], cfg["ppc"]
        p.adaptiveThreshWinSizeMin = wmin
        p.adaptiveThreshWinSizeMax = wmax
        p.adaptiveThreshWinSizeStep = wstep
        p.minMarkerPerimeterRate = perim
        p.maxMarkerPerimeterRate = 4.0
        p.polygonalApproxAccuracyRate = poly
        p.perspectiveRemovePixelPerCell = ppc
        p.perspectiveRemoveIgnoredMarginPerCell = 0.13
        _set_if_exists(p, "errorCorrectionRate", cfg["ecr"])

        if cfg["refine"] == "subpix" or roi:
            p.cornerRefinementMethod = a.CORNER_REFINE_SUBPIX
            p.cornerRefinementWinSize = 5
            p.cornerRefinementMaxIterations = 20
            _set_if_exists(p, "cornerRefinementMinAccuracy", 0.05)
        else:
            p.cornerRefinementMethod = a.CORNER_REFINE_NONE

        if cfg["aruco3"] and not roi:
            if _set_if_exists(p, "useAruco3Detection", True):
                _set_if_exists(p, "minSideLengthCanonicalImg", 16)
                _set_if_exists(p, "minMarkerLengthRatioOriginalImg", 0.0)
            else:
                log.info("useAruco3Detection absent de cette version d'OpenCV")
        return p

    def _get_roi_detect_fn(self, crop_w: int, crop_h: int) -> Callable:
        """Detecteur ROI adapte a la taille du crop, mis en cache par
        classe de taille (arrondie au multiple de 32 px superieur) : un
        ArucoDetector n'est reconstruit que pour une NOUVELLE classe de
        taille, jamais a chaque image. minMarkerPerimeterRate est derive de
        roi_min_marker_side_px (cote minimal absolu, pas relatif au crop —
        voir la note dans __init__)."""
        bw = ((crop_w + 31) // 32) * 32
        bh = ((crop_h + 31) // 32) * 32
        key = (bw, bh)
        fn = self._roi_cache.get(key)
        if fn is None:
            if len(self._roi_cache) >= 32:      # garde-fou anti-fuite memoire
                self._roi_cache.clear()
            perim = float(np.clip(4.0 * self.roi_min_marker_side_px / max(bw, bh),
                                  0.02, 0.5))
            params = self._build_params(self._preset_cfg, roi=True, perim_override=perim)
            fn = _make_detect_fn(self.dictionary, params)
            self._roi_cache[key] = fn
        return fn

    # ---- detection -------------------------------------------------------
    def _run(self, detect_fn, img: np.ndarray, ox: int = 0, oy: int = 0
             ) -> List["Detection"]:
        corners, ids, _ = detect_fn(img)
        if ids is None or len(ids) == 0:
            return []
        out: List["Detection"] = []
        off = np.float32((ox, oy)) if (ox or oy) else None
        for quad, mid in zip(corners, ids.reshape(-1)):
            q = quad.reshape(4, 2)
            q = (q + off) if off is not None else q.copy()
            out.append(Detection(int(mid), q))
        return out

    def detect(self, image: np.ndarray, t: Optional[float] = None):
        """Retourne (gardes, [(supprime, motif), ...]).

        image : plan Y (2D uint8) de preference. Une image BGR est convertie,
        mais avec --picam ce cas ne se produit jamais.
        t : horodatage de cette image, secondes, horloge monotone. Par
        defaut time.perf_counter() a l'entree de cet appel — suffisant pour
        estimer un dt reel entre detections (voir Track), mais si l'appelant
        dispose de l'horodatage materiel de la camera (SensorTimestamp
        libcamera, par exemple), le lui passer ici donne une prediction
        legerement plus fidele (elimine la gigue de traitement en amont).
        """
        now = t if t is not None else time.perf_counter()
        if self._last_call_t is not None:
            dt = now - self._last_call_t
            if dt > 0:
                self._ema_dt = 0.8 * self._ema_dt + 0.2 * dt
        self._last_call_t = now

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        h, w = gray.shape[:2]
        st = self.stats

        tracks = self.tracker.tracks
        full, why = (True, "roi desactive") if not self.roi_tracking \
            else self.policy.want_full(tracks, gray)

        rois: List[Tuple[int, int, int, int]] = []
        if not full:
            rois = self.planner.plan(tracks, w, h, expected_dt=self._ema_dt)
            if not rois:
                full, why = True, "roi trop larges"

        t0 = time.perf_counter()
        if full:
            n_before = len(tracks)
            raw = self._run(self._detect_full, gray)
            self.policy.after_full(len(raw), n_before)
            self.last_mode = "full"
            self.last_rois = 0
            st["full"] = st.get("full", 0) + 1
            log.debug("balayage complet (%s) -> %d candidats", why, len(raw))
        else:
            raw = self._detect_rois(gray, rois)
            ok = len(raw) >= len(tracks)
            self.last_mode = "roi"
            self.last_rois = len(rois)
            st["roi"] = st.get("roi", 0) + 1
            st["rois"] = st.get("rois", 0) + len(rois)

            if not ok:
                # RECUPERATION A PALIERS (remplace l'ancien passage direct a
                # un balayage complet) : un marqueur suivi qui manque a
                # l'appel signifie le plus souvent qu'il a un peu plus
                # bouge que prevu (vitesse sous-estimee, quelques images
                # sautees, rolling shutter) — pas qu'il a disparu. Une ROI
                # elargie coute une fraction du prix d'un plein cadre et
                # retrouve la grande majorite de ces cas ; le plein cadre
                # reste le dernier recours, jamais le premier reflexe.
                rois_wide = self.planner.plan(tracks, w, h, expected_dt=self._ema_dt,
                                              margin_scale=self.roi_retry_margin_scale)
                st["roi_retry"] = st.get("roi_retry", 0) + 1
                if rois_wide and rois_wide != rois:
                    raw_wide = self._detect_rois(gray, rois_wide)
                    if len(raw_wide) >= len(tracks):
                        raw = raw_wide
                        ok = True
                        self.last_mode = "roi_retry"
                        self.last_rois = len(rois_wide)
                        st["roi_retry_ok"] = st.get("roi_retry_ok", 0) + 1

            self.policy.after_roi(ok)
            if not ok:
                # Toujours perdu apres la ROI elargie : plein cadre.
                raw = self._run(self._detect_full, gray)
                self.policy.after_full(len(raw), len(tracks))
                self.last_mode = "roi+full"
                st["rescan"] = st.get("rescan", 0) + 1
        st["t_aruco"] = st.get("t_aruco", 0.0) + (time.perf_counter() - t0)

        self.last_raw = len(raw)
        st["raw"] = st.get("raw", 0) + len(raw)

        t0 = time.perf_counter()
        kept, removed = deduplicate(
            raw,
            score_fn=lambda d: self.scorer(gray, d),
            iou_thresh=self.iou_thresh,
            nested_thresh=self.nested_thresh,
            unique_ids=self.unique_ids,
            min_score=self.min_score,
            stats=st)
        st["t_dedup"] = st.get("t_dedup", 0.0) + (time.perf_counter() - t0)

        st["kept"] = st.get("kept", 0) + len(kept)
        st["removed"] = st.get("removed", 0) + len(removed)

        t0 = time.perf_counter()
        published = self.tracker.update(kept, now)
        st["t_track"] = st.get("t_track", 0.0) + (time.perf_counter() - t0)
        st["n_calls"] = st.get("n_calls", 0) + 1

        return (published if self.stabilize else kept), removed

    def _detect_rois(self, gray: np.ndarray,
                     rois: List[Tuple[int, int, int, int]]) -> List["Detection"]:
        raw: List["Detection"] = []
        for (x0, y0, x1, y1) in rois:
            crop = gray[y0:y1, x0:x1]
            if not crop.flags["C_CONTIGUOUS"]:
                crop = np.ascontiguousarray(crop)
            fn = self._get_roi_detect_fn(crop.shape[1], crop.shape[0])
            raw.extend(self._run(fn, crop, x0, y0))
        return raw

    def reset(self) -> None:
        self.tracker.tracks.clear()
        self.policy.force = True
        self._last_call_t = None
        self._ema_dt = 1.0 / 30.0


# ==========================================================================
# Sources d'images
# ==========================================================================
class FrameSource:
    dropped = 0

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @property
    def size(self) -> Tuple[int, int]:
        return (0, 0)


class PiCam3Source(FrameSource):
    """Camera Module 3 (IMX708) via Picamera2 — plan Y, triple buffer.

    Le thread de capture ecrit dans l'un de trois tampons prealloues et ne
    publie qu'une reference : le consommateur en detient un, un autre est
    "le plus recent", le troisieme est libre pour l'ecriture en cours. Une
    seule recopie par image (v1 : deux, copyto puis .copy()).

    Securite memoire : la recopie a lieu DANS le bloc `with MappedArray`,
    puis la requete est relachee. Aucun acces au buffer libcamera apres
    release().
    """

    SENSOR_MODES = {
        "fast": ((1536, 864), 120.0),      # binning 2x2, le mode utile ici
        "mid":  ((2304, 1296), 56.0),
        "full": ((4608, 2592), 14.0),
    }

    def __init__(self, width: int = 768, height: int = 432, fps: float = 120.0,
                 sensor_mode: str = "fast", exposure_us: int = 4000,
                 gain: float = 2.0, lens_position: Optional[float] = None,
                 buffer_count: int = 4, threaded: bool = True):
        try:
            from picamera2 import Picamera2, MappedArray
        except ImportError as e:
            raise RuntimeError(
                "picamera2 absent. Sur Raspberry Pi OS :\n"
                "    sudo apt install -y python3-picamera2 python3-opencv") from e
        self._MappedArray = MappedArray

        # YUV420 : largeur multiple de 16, hauteur multiple de 2.
        self.w = max(64, (int(width) // 16) * 16)
        self.h = max(64, (int(height) // 2) * 2)

        sensor_size, mode_fps = self.SENSOR_MODES.get(
            sensor_mode, self.SENSOR_MODES["fast"])
        # [FIABILITE] Un rapport d'aspect different de celui du mode capteur
        # fait etirer l'image par l'ISP : les marqueurs ne sont plus carres
        # et toute pose calculee ensuite est fausse.
        ar_out = self.w / self.h
        ar_sensor = sensor_size[0] / sensor_size[1]
        if abs(ar_out - ar_sensor) / ar_sensor > 0.02:
            log.warning(
                "rapport d'aspect %.2f different du mode capteur %.2f : l'ISP "
                "etire l'image (pixels non carres). Preferer %dx%d ou definir "
                "un ScalerCrop.", ar_out, ar_sensor,
                self.w, int(round(self.w / ar_sensor / 2) * 2))

        if fps > mode_fps:
            log.warning("%.0f fps demandes, mode capteur %s limite a %.0f",
                        fps, sensor_mode, mode_fps)
            fps = mode_fps
        frame_us = int(1_000_000 / max(fps, 1.0))
        exposure_us = int(min(exposure_us, frame_us - 200))
        if exposure_us < 100:
            raise ValueError("temps de pose trop court pour cette frequence")

        self.picam2 = Picamera2()
        ctrls = {
            "FrameDurationLimits": (frame_us, frame_us),
            "AeEnable": False,          # pose constante -> flou de bouge constant
            "AwbEnable": False,         # inutile en mono, et evite un calcul ISP
            "ExposureTime": exposure_us,
            "AnalogueGain": float(gain),
        }
        cfg = None
        try:
            cfg = self.picam2.create_video_configuration(
                main={"size": (self.w, self.h), "format": "YUV420"},
                sensor={"output_size": sensor_size, "bit_depth": 10},
                controls=ctrls, buffer_count=buffer_count, queue=False)
        except TypeError:
            log.info("picamera2 sans argument 'sensor' : repli sur raw=")
        if cfg is None:
            cfg = self.picam2.create_video_configuration(
                main={"size": (self.w, self.h), "format": "YUV420"},
                raw={"size": sensor_size},
                controls=ctrls, buffer_count=buffer_count, queue=False)
        self.picam2.configure(cfg)
        self._apply_optional_controls(lens_position)
        self.picam2.start()
        time.sleep(0.25)                                 # remplissage du pipeline

        mc = self.picam2.camera_configuration()["main"]
        self.w, self.h = mc["size"]
        self.stride = mc["stride"]
        log.info("flux ISP %dx%d stride=%d, mode capteur %s, pose %d us, gain %.1f",
                 self.w, self.h, self.stride, sensor_size, exposure_us, gain)

        self._bufs = [np.empty((self.h, self.w), dtype=np.uint8) for _ in range(3)]
        self._free: List[int] = [0, 1, 2]
        self._latest: Optional[int] = None
        self._held: Optional[int] = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        self.dropped = 0
        self.error: Optional[BaseException] = None
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._loop, name="picam",
                                            daemon=True)
            self._thread.start()

    def _apply_optional_controls(self, lens_position) -> None:
        opt: Dict[str, object] = {}
        try:
            from libcamera import controls as lc
            try:
                # Le debruitage ISP coute du temps et arrondit les coins ;
                # certains empilements n'acceptent que Minimal.
                opt["NoiseReductionMode"] = lc.draft.NoiseReductionModeEnum.Off
            except AttributeError:
                log.debug("NoiseReductionMode indisponible")
            # Le Module 3 est le seul module Pi avec autofocus : le CAF
            # chasse en permanence et floute les coins. Manuel obligatoire.
            opt["AfMode"] = lc.AfModeEnum.Manual
            if lens_position is not None:
                opt["LensPosition"] = float(lens_position)   # dioptries = 1/Z
        except ImportError:
            log.warning("libcamera non importable : AF/debruitage laisses par defaut")
        opt["Sharpness"] = 1.0
        for k, v in opt.items():
            try:
                self.picam2.set_controls({k: v})
            except Exception as e:                        # [ROBUSTESSE] tracee
                log.warning("controle %s refuse par le pipeline : %s", k, e)

    def _grab_into(self, idx: int) -> None:
        req = self.picam2.capture_request()
        try:
            with self._MappedArray(req, "main") as m:
                arr = m.array
                if arr.ndim == 2 and arr.shape[1] >= self.w:
                    np.copyto(self._bufs[idx], arr[:self.h, :self.w])
                else:                                     # forme inattendue
                    flat = np.asarray(arr).reshape(-1)
                    np.copyto(self._bufs[idx],
                              flat[:self.stride * self.h]
                              .reshape(self.h, self.stride)[:, :self.w])
        finally:
            req.release()          # jamais d'acces au buffer libcamera ensuite

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                idx = self._free.pop() if self._free else None
            if idx is None:                              # ne doit pas arriver
                time.sleep(0.001)
                continue
            try:
                self._grab_into(idx)
            except Exception as e:
                self.error = e
                log.error("capture interrompue : %s", e)
                self._stop.set()
                self._new.set()
                return
            with self._lock:
                if self._latest is not None:
                    self._free.append(self._latest)      # latest-wins
                    self.dropped += 1
                self._latest = idx
            self._new.set()

    def read(self) -> Optional[np.ndarray]:
        if self._thread is None:
            with self._lock:
                idx = self._free.pop()
            self._grab_into(idx)
            with self._lock:
                if self._held is not None:
                    self._free.append(self._held)
                self._held = idx
            return self._bufs[idx]

        if not self._new.wait(timeout=2.0):
            log.error("aucune image depuis 2 s")
            return None
        with self._lock:
            self._new.clear()
            if self._latest is None:
                return None if self._stop.is_set() else self.read()
            if self._held is not None:
                self._free.append(self._held)
            self._held, self._latest = self._latest, None
            idx = self._held
        if self.error is not None:
            raise self.error
        return self._bufs[idx]

    def close(self) -> None:
        self._stop.set()
        self._new.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception as e:
            log.warning("fermeture camera : %s", e)

    @property
    def size(self) -> Tuple[int, int]:
        return (self.w, self.h)


class CvSource(FrameSource):
    """Fichier video ou camera USB (repli hors Raspberry Pi)."""

    def __init__(self, src, width: Optional[int] = None,
                 height: Optional[int] = None, gray: bool = True):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"flux inaccessible : {src}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.gray = gray
        self._buf: Optional[np.ndarray] = None

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        if self.gray and frame.ndim == 3:
            # dst prealloue : cvtColor n'alloue plus a chaque image
            h, w = frame.shape[:2]
            if self._buf is None or self._buf.shape != (h, w):
                self._buf = np.empty((h, w), np.uint8)
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self._buf)
            return self._buf
        return frame

    def close(self) -> None:
        self.cap.release()

    @property
    def size(self) -> Tuple[int, int]:
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


# ==========================================================================
# Sortie video (thread separe)
# ==========================================================================
class AsyncWriter:
    """Encodage dans un thread, file bornee.

    VideoWriter fait l'encodage en synchrone dans le thread appelant : sur
    CM4, mp4v a 720p coute plusieurs millisecondes par image et deviendrait
    le facteur limitant. Ici l'encodeur prend du retard sans ralentir la
    detection ; les images en trop sont abandonnees et comptees.
    """

    def __init__(self, path: str, size: Tuple[int, int], fps: float,
                 fourcc: str = "mp4v", maxsize: int = 3):
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc),
                                      max(1.0, min(fps, 120.0)), size)
        if not self.writer.isOpened():
            raise RuntimeError(f"VideoWriter : impossible d'ouvrir {path}")
        self.q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self._t = threading.Thread(target=self._loop, name="writer", daemon=True)
        self._t.start()

    def _loop(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                self.writer.write(item)
            except Exception as e:                        # [ROBUSTESSE]
                log.error("ecriture video : %s", e)
                return

    def submit(self, frame: np.ndarray) -> None:
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self._t.join(timeout=3.0)
        self.writer.release()


# ==========================================================================
# Affichage / rapport
# ==========================================================================
def draw(image: np.ndarray, kept, removed, show_removed: bool = True,
         hud: str = "", dst: Optional[np.ndarray] = None) -> np.ndarray:
    """Rendu annote. dst prealloue evite une allocation BGR par image."""
    if image.ndim == 2:
        if dst is None or dst.shape[:2] != image.shape:
            dst = np.empty((image.shape[0], image.shape[1], 3), np.uint8)
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR, dst=dst)
        out = dst
    else:
        out = image.copy()

    if show_removed:
        for d, reason in removed:
            pts = d.corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, (0, 0, 255), 1)
            cv2.putText(out, reason, (int(d.corners[0][0]), int(d.corners[0][1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for d in kept:
        pts = d.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 220, 0), 2)
        cv2.circle(out, (int(d.corners[0][0]), int(d.corners[0][1])), 4, (255, 0, 0), -1)
        label = f"#{d.marker_id}" + (f" q={d.score:.2f}" if d.score is not None else "")
        cv2.putText(out, label, (int(d.cx) - 28, int(d.cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)
    if hud:
        cv2.putText(out, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
    return out


def report(kept, removed) -> str:
    lines = ["[OK] %d marqueur(s) : " % len(kept)
             + ", ".join(f"#{d.marker_id}"
                         + (f"(q={d.score:.2f})" if d.score is not None else "")
                         for d in kept)]
    if removed:
        lines.append(f"[--] {len(removed)} doublon(s) supprime(s) :")
        for d, reason in removed:
            lines.append(f"     #{d.marker_id} aire={d.area:.0f}px2 -> {reason}")
    return "\n".join(lines)


# ==========================================================================
# Instrumentation
# ==========================================================================
STAGES = ("capture", "detect", "dedup", "track", "display", "write")


class Stats:
    """Chronometrage par etage + percentiles. Le cout par image est de
    quelques appels perf_counter et un append dans une liste."""

    def __init__(self, keep: int = 20000):
        self.keep = keep
        self.t: Dict[str, List[float]] = {s: [] for s in STAGES}
        self.total: List[float] = []
        self.frames = 0
        self.t0 = time.perf_counter()
        self._cur: Dict[str, float] = {}

    def start(self, stage: str) -> float:
        return time.perf_counter()

    def stop(self, stage: str, t_start: float) -> None:
        self._cur[stage] = time.perf_counter() - t_start

    def commit(self) -> None:
        tot = 0.0
        for s in STAGES:
            v = self._cur.get(s, 0.0)
            tot += v
            lst = self.t[s]
            if len(lst) < self.keep:
                lst.append(v)
        if len(self.total) < self.keep:
            self.total.append(tot)
        self._cur.clear()
        self.frames += 1

    @staticmethod
    def _pct(v: List[float], q: float) -> float:
        return float(np.percentile(v, q)) if v else 0.0

    def hud(self) -> str:
        if not self.total:
            return ""
        recent = self.total[-60:]
        fps = 1.0 / max(sum(recent) / len(recent), 1e-9)
        det = self.t["detect"][-60:]
        return (f"{fps:5.1f} fps | det {1000*sum(det)/max(len(det),1):4.1f} ms")

    def profile_line(self) -> str:
        parts = []
        for s in STAGES:
            v = self.t[s][-120:]
            if v and sum(v) > 0:
                parts.append(f"{s} {1000*sum(v)/len(v):5.2f}")
        v = self.total[-120:]
        parts.append(f"total {1000*sum(v)/max(len(v),1):5.2f} ms")
        return " | ".join(parts)

    def summary(self, det_stats: Dict[str, int], src_dropped: int = 0,
                writer_dropped: int = 0) -> str:
        wall = time.perf_counter() - self.t0
        f = max(self.frames, 1)
        out = [f"{self.frames} images en {wall:.2f} s -> "
               f"{self.frames / max(wall, 1e-9):.1f} fps effectifs", "",
               "  etage         moy      p50      p95      p99"]
        for s in STAGES + ("total",):
            v = self.t[s] if s in self.t else self.total
            if not v or sum(v) <= 0:
                continue
            out.append(f"  {s:<10} {1000*sum(v)/len(v):7.2f}  "
                       f"{1000*self._pct(v,50):7.2f}  {1000*self._pct(v,95):7.2f}  "
                       f"{1000*self._pct(v,99):7.2f}   (ms)")
        n_calls = max(det_stats.get("n_calls", 0), 1)
        out += [
            "",
            f"  balayages complets   : {det_stats.get('full', 0)}",
            f"  balayages ROI        : {det_stats.get('roi', 0)}"
            f" ({det_stats.get('rois', 0)} fenetres,"
            f" {det_stats.get('roi_retry', 0)} recuperations tentees,"
            f" {det_stats.get('roi_retry_ok', 0)} reussies,"
            f" {det_stats.get('rescan', 0)} rebalayages complets par defaut)",
            f"  candidats bruts      : {det_stats.get('raw', 0)}"
            f" ({det_stats.get('raw', 0)/f:.2f}/image)",
            f"  marqueurs conserves  : {det_stats.get('kept', 0)}"
            f" ({det_stats.get('kept', 0)/f:.2f}/image)",
            f"  doublons supprimes   : {det_stats.get('removed', 0)}"
            f" ({det_stats.get('removed', 0)/f:.2f}/image)",
            f"  paires testees / scores calcules : "
            f"{det_stats.get('pairs', 0)} / {det_stats.get('scored', 0)}",
            "",
            "  detail interne a detect() (moyenne par appel, ms) :",
            f"    aruco (detectMarkers) : {1000*det_stats.get('t_aruco',0.0)/n_calls:7.3f}",
            f"    dedup                  : {1000*det_stats.get('t_dedup',0.0)/n_calls:7.3f}",
            f"    tracking                : {1000*det_stats.get('t_track',0.0)/n_calls:7.3f}",
            f"  images camera abandonnees : {src_dropped}",
        ]
        if writer_dropped:
            out.append(f"  images non encodees  : {writer_dropped}")
        return "\n".join(out)


# ==========================================================================
# CLI
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detection ArUco sans doublons, faible latence (CM4 + IMX708)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--picam", action="store_true", help="Camera Module 3 (Picamera2)")
    src.add_argument("--image")
    src.add_argument("--video")
    src.add_argument("--camera", type=int, help="camera USB via OpenCV")

    g = p.add_argument_group("marqueurs")
    g.add_argument("--dict", default="DICT_4X4_50")
    g.add_argument("--custom-dict", help="dictionnaire maison .npy (N, k, k)")
    g.add_argument("--preset", default="balanced", choices=sorted(PRESETS))
    g.add_argument("--unique-ids", action="store_true",
                   help="un seul marqueur par ID dans l'image")
    g.add_argument("--iou", type=float, default=0.35)
    g.add_argument("--nested", type=float, default=0.75)
    g.add_argument("--min-score", type=float, default=None)
    g.add_argument("--max-correction", type=int, default=None,
                   help="maxCorrectionBits (defaut : valeur du dictionnaire)")

    g = p.add_argument_group("suivi")
    g.add_argument("--no-roi", action="store_true", help="desactive le suivi par ROI")
    g.add_argument("--full-scan-every", type=int, default=32,
                   help="intervalle MAXIMAL entre deux balayages complets")
    g.add_argument("--roi-coverage", type=float, default=0.35,
                   help="surface max des ROI avant balayage direct")
    g.add_argument("--roi-margin", type=float, default=0.45)
    g.add_argument("--roi-min-side-px", type=float, default=12.0,
                   help="cote minimal ABSOLU accepte en ROI (remplace l'ancien "
                        "seuil relatif au crop, corrige — voir __init__)")
    g.add_argument("--roi-retry-scale", type=float, default=1.7,
                   help="facteur d'elargissement de la ROI avant de recourir "
                        "a un balayage complet")
    g.add_argument("--track-max-miss", type=int, default=3)
    g.add_argument("--stabilize", action="store_true",
                   help="publie les marqueurs lisses et persistants")

    g = p.add_argument_group("camera")
    g.add_argument("--width", type=int, default=768)
    g.add_argument("--height", type=int, default=432)
    g.add_argument("--fps", type=float, default=120.0)
    g.add_argument("--sensor-mode", default="fast",
                   choices=sorted(PiCam3Source.SENSOR_MODES))
    g.add_argument("--exposure", type=int, default=4000, help="temps de pose (us)")
    g.add_argument("--gain", type=float, default=2.0)
    g.add_argument("--lens-position", type=float, default=None,
                   help="mise au point manuelle en dioptries (0 = infini)")

    g = p.add_argument_group("sortie")
    g.add_argument("--threads", type=int, default=3, help="threads OpenCV")
    g.add_argument("--out", help="fichier video annote")
    g.add_argument("--report", action="store_true")
    g.add_argument("--bench", type=int, default=0, help="mesurer N images puis sortir")
    g.add_argument("--display", action="store_true")
    g.add_argument("--no-display", action="store_true")
    g.add_argument("--display-every", type=int, default=1,
                   help="n'afficher qu'une image sur N")
    g.add_argument("--profile", action="store_true",
                   help="temps par etage toutes les 60 images")
    g.add_argument("--debug", action="store_true")
    return p


def run_image(a, det: ArucoCleanDetector) -> int:
    img = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.error("image illisible ou format non supporte : %s", a.image)
        return 1
    det.roi_tracking = False
    kept, removed = det.detect(img)
    print(report(kept, removed))
    if a.out:
        if not cv2.imwrite(a.out, draw(img, kept, removed)):
            log.error("ecriture impossible : %s", a.out)
            return 1
    if a.display and not a.no_display:
        cv2.imshow("aruco_dedup_v2", draw(img, kept, removed))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_stream(a, det: ArucoCleanDetector) -> int:
    if a.picam:
        source: FrameSource = PiCam3Source(
            width=a.width, height=a.height, fps=a.fps, sensor_mode=a.sensor_mode,
            exposure_us=a.exposure, gain=a.gain, lens_position=a.lens_position)
        print(f"Camera Module 3 : {source.size[0]}x{source.size[1]} "
              f"@ {a.fps:g} fps vises, mode capteur {a.sensor_mode}")
    else:
        source = CvSource(a.video if a.video else a.camera, a.width, a.height)

    show = a.display and not a.no_display
    writer: Optional[AsyncWriter] = None
    stats = Stats()
    vis_buf: Optional[np.ndarray] = None
    rc = 0

    try:
        while True:
            t = stats.start("capture")
            frame = source.read()
            stats.stop("capture", t)
            if frame is None:
                break

            t = stats.start("detect")
            kept, removed = det.detect(frame)
            stats.stop("detect", t)

            if writer is None and a.out:
                h, w = frame.shape[:2]
                writer = AsyncWriter(a.out, (w, h), min(a.fps, 60.0))

            if show or writer is not None:
                t = stats.start("display")
                vis = draw(frame, kept, removed, hud=stats.hud(), dst=vis_buf)
                vis_buf = vis
                stats.stop("display", t)
                if writer is not None:
                    t = stats.start("write")
                    writer.submit(vis.copy())
                    stats.stop("write", t)
                if show and stats.frames % max(1, a.display_every) == 0:
                    cv2.imshow("aruco_dedup_v2", vis)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

            stats.commit()
            if a.profile and stats.frames % 60 == 0:
                print(stats.profile_line())
            elif a.report and stats.frames % 60 == 0:
                print(f"{stats.hud()} | mode={det.last_mode} "
                      f"bruts={det.last_raw} gardes={len(kept)}")
            if a.bench and stats.frames >= a.bench:
                break
    except KeyboardInterrupt:
        print()
    except RuntimeError as e:
        log.error("%s", e)
        rc = 1
    finally:
        source.close()
        if writer is not None:
            writer.close()
        if show:
            cv2.destroyAllWindows()

    if stats.frames:
        print(stats.summary(det.stats, getattr(source, "dropped", 0),
                            writer.dropped if writer else 0))
    elif rc == 0:
        log.error("aucune image lue")
        rc = 1
    return rc


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.debug else logging.INFO,
                        format="%(levelname)s %(message)s")

    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, a.threads))

    try:
        det = ArucoCleanDetector(
            dict_name=a.dict, custom_dict=a.custom_dict, preset=a.preset,
            unique_ids=a.unique_ids, iou_thresh=a.iou, nested_thresh=a.nested,
            min_score=a.min_score, stabilize=a.stabilize,
            roi_tracking=not a.no_roi, full_scan_every=a.full_scan_every,
            roi_margin=a.roi_margin, roi_max_coverage=a.roi_coverage,
            roi_min_marker_side_px=a.roi_min_side_px,
            roi_retry_margin_scale=a.roi_retry_scale,
            track_max_miss=a.track_max_miss, max_correction=a.max_correction)
    except (ValueError, RuntimeError, OSError) as e:
        log.error("%s", e)
        return 2

    try:
        if a.image:
            return run_image(a, det)
        return run_stream(a, det)
    except RuntimeError as e:                              # camera absente, etc.
        log.error("%s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
