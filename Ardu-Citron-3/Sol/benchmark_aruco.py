#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_aruco.py
==================

Benchmark complet du détecteur ArUco actuel + générateur de dataset réaliste.

Objectifs :
  1. Générer UNE fois un dataset réaliste et le réutiliser pour toutes les
     configurations afin que les comparaisons soient équitables.
  2. Mesurer vitesse + latence + p50/p95/p99 + FPS équivalent.
  3. Mesurer précision : recall, precision, F1, faux positifs, faux négatifs,
     mauvais IDs, localisation, erreur de coins, IoU.
  4. Mesurer le coût interne : ArUco, dedup, tracking.
  5. Mesurer le comportement ROI / full scan / retry / rescan.
  6. Produire CSV + JSON + Markdown + éventuellement graphiques.
  7. Faire en plus un benchmark temporel "synthétique" à partir des mêmes
     images, pour réellement solliciter le tracking ROI.

IMPORTANT :
  - Le benchmark statique remet le tracker à zéro pour chaque image : il
    mesure la qualité intrinsèque image -> détection.
  - Le benchmark temporel applique de petites transformations géométriques
    déterministes à chaque image de base. Il sert à comparer le comportement
    ROI/tracking, mais ce n'est PAS une simulation physique complète d'une
    trajectoire.
  - Les scores de précision utilisent la vérité-terrain JSON produite par
    generate_realistic_dataset.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Utilitaires fichiers / modules
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger le module : {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_default_file(pattern: str) -> Optional[Path]:
    roots = [Path(__file__).resolve().parent, Path.cwd()]
    candidates = []
    seen = set()
    for root in roots:
        for p in root.glob(pattern):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                candidates.append(rp)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def json_dump(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean_or_zero(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def run_generator(
    generator: Path,
    dataset_root: Path,
    n_scenes: int,
    markers_per_scene: str,
    marker_ids: str,
    difficulty: str,
    seed: int,
    marker_size_m: Optional[float],
) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)

    difficulties = ["normal", "hard"] if difficulty == "both" else [difficulty]

    for diff in difficulties:
        out = dataset_root / diff if len(difficulties) > 1 else dataset_root
        if out.exists():
            out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(generator),
            "--n", str(n_scenes),
            "--out", str(out),
            "--markers-per-scene", markers_per_scene,
            "--marker-ids", marker_ids,
            "--difficulty", diff,
            "--seed", str(seed if diff == "normal" else seed + 100003),
        ]
        if marker_size_m is not None:
            cmd += ["--marker-size-m", str(marker_size_m)]

        print("\n[DATASET] génération :", " ".join(map(str, cmd)))
        rc = subprocess.run(cmd)
        if rc.returncode != 0:
            raise SystemExit(f"Génération du dataset échouée pour {diff} (code {rc.returncode}).")


def dataset_files(root: Path, max_images: int = 0) -> List[Tuple[Path, Path, str]]:
    items: List[Tuple[Path, Path, str]] = []
    dirs = [root / "normal", root / "hard"] if (root / "normal").exists() else [root]

    for d in dirs:
        for js in sorted(d.glob("scene_*.json")):
            img = js.with_suffix(".png")
            if not img.exists():
                continue
            difficulty = d.name if d.name in ("normal", "hard") else "normal"
            items.append((img, js, difficulty))

    if max_images > 0:
        items = items[:max_images]
    return items


def load_scene(json_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("markers", []), d.get("scene", {})


# ---------------------------------------------------------------------------
# Géométrie / matching
# ---------------------------------------------------------------------------

def polygon_area(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return float(abs(cv2.contourArea(p)))


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(4, 2)
    b = np.asarray(b, dtype=np.float32).reshape(4, 2)
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
    except cv2.error:
        return 0.0
    inter = float(inter)
    if inter <= 0.0:
        return 0.0
    aa = polygon_area(a)
    bb = polygon_area(b)
    union = aa + bb - inter
    return inter / union if union > 1e-9 else 0.0


def corners_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    RMSE des coins en prenant la meilleure rotation cyclique et la meilleure
    orientation. Cela évite qu'un simple décalage d'index des quatre coins
    transforme une bonne localisation en grosse erreur.
    """
    a = np.asarray(a, dtype=np.float32).reshape(4, 2)
    b = np.asarray(b, dtype=np.float32).reshape(4, 2)
    variants = []
    for k in range(4):
        variants.append(np.roll(a, k, axis=0))
    ar = a[::-1]
    for k in range(4):
        variants.append(np.roll(ar, k, axis=0))

    best = float("inf")
    for v in variants:
        err = np.sqrt(np.mean(np.sum((v - b) ** 2, axis=1)))
        best = min(best, float(err))
    return best


def quad_center(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float32).reshape(4, 2).mean(axis=0)


def transform_corners(corners: np.ndarray, M: np.ndarray) -> np.ndarray:
    p = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    ones = np.ones((4, 1), np.float32)
    ph = np.hstack([p, ones])
    return (ph @ M.T).astype(np.float32)


def match_predictions(
    gt_markers: List[Dict[str, Any]],
    predictions: Sequence[Any],
    iou_threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Matching géométrique un-à-un, puis classification ID correcte / mauvaise.

    Un candidat ayant une bonne correspondance géométrique mais un mauvais ID
    est compté comme "wrong_id", pas comme simple faux positif.
    """
    gt_quads = [np.asarray(m["corners_px"], dtype=np.float32) for m in gt_markers]
    pred_quads = [np.asarray(d.corners, dtype=np.float32) for d in predictions]
    pred_ids = [int(d.marker_id) for d in predictions]
    gt_ids = [int(m["marker_id"]) for m in gt_markers]

    pairs: List[Tuple[float, int, int]] = []
    for pi, pq in enumerate(pred_quads):
        for gi, gq in enumerate(gt_quads):
            iou = polygon_iou(pq, gq)
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)

    used_p = set()
    used_g = set()
    matches = []
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, iou))

    exact = 0
    wrong_id = 0
    loc_errors = []
    center_errors = []
    ious = []
    match_details = []

    for pi, gi, iou in matches:
        exact_id = pred_ids[pi] == gt_ids[gi]
        exact += int(exact_id)
        wrong_id += int(not exact_id)

        rmse = corners_rmse(pred_quads[pi], gt_quads[gi])
        ce = float(np.linalg.norm(quad_center(pred_quads[pi]) -
                                 quad_center(gt_quads[gi])))
        loc_errors.append(rmse)
        center_errors.append(ce)
        ious.append(iou)

        match_details.append({
            "pred_index": pi,
            "gt_index": gi,
            "pred_id": pred_ids[pi],
            "gt_id": gt_ids[gi],
            "iou": float(iou),
            "corner_rmse_px": float(rmse),
            "center_error_px": float(ce),
            "id_ok": bool(exact_id),
        })

    fp = len(predictions) - len(matches)
    fn = len(gt_markers) - len(matches)

    return {
        "pred_count": len(predictions),
        "gt_count": len(gt_markers),
        "localized": len(matches),
        "correct_id": exact,
        "wrong_id": wrong_id,
        "fp": fp,
        "fn": fn,
        "matches": match_details,
        "mean_iou": mean_or_zero(ious),
        "mean_corner_rmse_px": mean_or_zero(loc_errors),
        "mean_center_error_px": mean_or_zero(center_errors),
    }


# ---------------------------------------------------------------------------
# Statistiques internes du détecteur
# ---------------------------------------------------------------------------

def stat_delta(before: Dict[str, Any], after: Dict[str, Any], key: str, default=0):
    a = before.get(key, default)
    b = after.get(key, default)
    return b - a


def snapshot_stats(det) -> Dict[str, Any]:
    return dict(getattr(det, "stats", {}))


# ---------------------------------------------------------------------------
# Transformations temporelles synthétiques
# ---------------------------------------------------------------------------

def temporal_transform(frame_idx: int, width: int, height: int) -> np.ndarray:
    """
    Petite trajectoire artificielle, volontairement déterministe :
      - translation progressive
      - rotation légère
      - variation d'échelle légère

    L'objectif n'est pas la physique mais le déclenchement du tracking/ROI.
    """
    seq = [
        (0.00, 1.000, 0.0, 0.0),
        (0.7,  1.010, 5.0, 2.0),
        (1.1,  1.018, 10.0, 4.0),
        (-0.8, 1.012, 15.0, 6.0),
        (-1.4, 1.025, 20.0, 7.0),
        (0.5,  1.035, 24.0, 9.0),
        (1.3,  1.045, 28.0, 10.0),
        (0.0,  1.055, 32.0, 11.0),
    ]
    angle, scale, tx, ty = seq[min(frame_idx, len(seq) - 1)]
    M2 = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    M2[:, 2] += (tx, ty)
    return M2.astype(np.float32)


def transformed_scene(
    image: np.ndarray,
    gt_markers: List[Dict[str, Any]],
    frame_idx: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    h, w = image.shape[:2]
    M = temporal_transform(frame_idx, w, h)
    out = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    gt2 = []
    for m in gt_markers:
        mm = dict(m)
        mm["corners_px"] = transform_corners(
            np.asarray(m["corners_px"], dtype=np.float32), M
        ).tolist()
        xs = np.asarray(mm["corners_px"], dtype=np.float32)[:, 0]
        ys = np.asarray(mm["corners_px"], dtype=np.float32)[:, 1]
        mm["marker_side_px"] = float(
            0.5 * (np.linalg.norm(np.array(mm["corners_px"][0]) -
                                  np.array(mm["corners_px"][1])) +
                   np.linalg.norm(np.array(mm["corners_px"][1]) -
                                  np.array(mm["corners_px"][2])))
        )
        mm["_visible_bbox"] = [
            float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        ]
        gt2.append(mm)
    return out, gt2


# ---------------------------------------------------------------------------
# Variantes à tester
# ---------------------------------------------------------------------------

def build_variants(args) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []

    # Base : exactement les trois presets fournis par le code actuel.
    presets = [x.strip() for x in args.presets.split(",") if x.strip()]
    for p in presets:
        variants.append({
            "name": f"{p}_roi",
            "preset": p,
            "roi": True,
            "full_scan_every": 32,
            "roi_min_side": 12.0,
            "threads": args.threads,
        })

    if args.include_no_roi:
        for p in presets:
            variants.append({
                "name": f"{p}_full_only",
                "preset": p,
                "roi": False,
                "full_scan_every": 32,
                "roi_min_side": 12.0,
                "threads": args.threads,
            })

    if args.sweep:
        # Variantes ciblées sur les leviers les plus importants de la v2.1.
        for side in (12.0, 16.0, 20.0):
            variants.append({
                "name": f"balanced_roi_side{int(side)}",
                "preset": "balanced",
                "roi": True,
                "full_scan_every": 32,
                "roi_min_side": side,
                "threads": args.threads,
            })
        for every in (8, 16, 32):
            variants.append({
                "name": f"balanced_roi_full{every}",
                "preset": "balanced",
                "roi": True,
                "full_scan_every": every,
                "roi_min_side": 12.0,
                "threads": args.threads,
            })

    if args.sweep_threads:
        for th in (1, 2, 3, 4):
            variants.append({
                "name": f"balanced_roi_thr{th}",
                "preset": "balanced",
                "roi": True,
                "full_scan_every": 32,
                "roi_min_side": 12.0,
                "threads": th,
            })

    # Évite les doublons si sweep reproduit une variante déjà créée.
    uniq = {}
    for v in variants:
        uniq[v["name"]] = v
    return list(uniq.values())


def make_detector(mod, v: Dict[str, Any], args):
    cv2.setNumThreads(int(v["threads"]))
    det = mod.ArucoCleanDetector(
        dict_name=args.dictionary,
        preset=v["preset"],
        unique_ids=args.unique_ids,
        roi_tracking=v["roi"],
        full_scan_every=int(v["full_scan_every"]),
        roi_min_marker_side_px=float(v["roi_min_side"]),
        roi_retry_margin_scale=float(args.roi_retry_scale),
        track_max_miss=int(args.track_max_miss),
        min_score=args.min_score,
    )
    return det


# ---------------------------------------------------------------------------
# Exécution d'un cas
# ---------------------------------------------------------------------------

def benchmark_static(
    mod,
    variant: Dict[str, Any],
    items: List[Tuple[Path, Path, str]],
    args,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    det = make_detector(mod, variant, args)
    rows = []

    print(f"  [STATIC] {variant['name']} : {len(items)} images")

    # Warmup sur quelques images, non compté.
    warmup_n = min(args.warmup, len(items))
    for i in range(warmup_n):
        img = cv2.imread(str(items[i][0]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            det.reset()
            det.detect(img, t=0.0)

    for idx, (img_path, json_path, difficulty) in enumerate(items):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        gt, scene = load_scene(json_path)

        # Une scène = une image indépendante : reset volontaire.
        det.reset()

        s0 = snapshot_stats(det)
        t0 = time.perf_counter()
        preds, removed = det.detect(img, t=float(idx) / 30.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        s1 = snapshot_stats(det)

        m = match_predictions(gt, preds, args.match_iou)

        row = {
            "variant": variant["name"],
            "benchmark": "static",
            "image": img_path.name,
            "difficulty": difficulty,
            "ground_truth": len(gt),
            "predictions": len(preds),
            "localized": m["localized"],
            "correct_id": m["correct_id"],
            "wrong_id": m["wrong_id"],
            "false_positive": m["fp"],
            "false_negative": m["fn"],
            "localization_precision": (
                m["localized"] / max(len(preds), 1)
            ),
            "localization_recall": (
                m["localized"] / max(len(gt), 1)
            ),
            "id_precision": (
                m["correct_id"] / max(len(preds), 1)
            ),
            "id_recall": (
                m["correct_id"] / max(len(gt), 1)
            ),
            "mean_iou": m["mean_iou"],
            "corner_rmse_px": m["mean_corner_rmse_px"],
            "center_error_px": m["mean_center_error_px"],
            "latency_ms": elapsed_ms,
            "aruco_ms": 1000.0 * stat_delta(s0, s1, "t_aruco", 0.0),
            "dedup_ms": 1000.0 * stat_delta(s0, s1, "t_dedup", 0.0),
            "track_ms": 1000.0 * stat_delta(s0, s1, "t_track", 0.0),
            "full_calls": stat_delta(s0, s1, "full", 0),
            "roi_calls": stat_delta(s0, s1, "roi", 0),
            "roi_retry": stat_delta(s0, s1, "roi_retry", 0),
            "roi_retry_ok": stat_delta(s0, s1, "roi_retry_ok", 0),
            "rescan": stat_delta(s0, s1, "rescan", 0),
            "raw_candidates": stat_delta(s0, s1, "raw", 0),
            "kept": stat_delta(s0, s1, "kept", 0),
            "removed": stat_delta(s0, s1, "removed", 0),
            "scene_texture": scene.get("ground_texture", ""),
            "altitude_m": scene.get("altitude_ref_m", None),
            "motion_blur_px": scene.get("motion_blur_len_px", None),
            "defocus_ksize": scene.get("defocus_blur_ksize", None),
            "vibration_px": scene.get("vibration_px", None),
            "visibility_mean": mean_or_zero(
                [float(x.get("visible_fraction", 0.0)) for x in gt]
            ),
            "marker_side_mean_px": mean_or_zero(
                [float(x.get("marker_side_px", 0.0)) for x in gt]
            ),
        }

        rows.append(row)

        if args.progress and ((idx + 1) % max(1, args.progress) == 0):
            print(f"    {idx+1}/{len(items)}")

    return rows, aggregate_variant(rows, variant["name"], "static")


def benchmark_temporal(
    mod,
    variant: Dict[str, Any],
    items: List[Tuple[Path, Path, str]],
    args,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    det = make_detector(mod, variant, args)
    rows = []

    n_items = min(args.temporal_scenes, len(items))
    items = items[:n_items]
    print(
        f"  [TEMPORAL] {variant['name']} : "
        f"{n_items} séquences × {args.temporal_frames} frames"
    )

    for seq_idx, (img_path, json_path, difficulty) in enumerate(items):
        base = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if base is None:
            continue
        gt0, scene = load_scene(json_path)
        det.reset()

        for fi in range(args.temporal_frames):
            img, gt = transformed_scene(base, gt0, fi)

            s0 = snapshot_stats(det)
            t0 = time.perf_counter()
            preds, removed = det.detect(img, t=float(fi) / args.temporal_fps)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            s1 = snapshot_stats(det)

            m = match_predictions(gt, preds, args.match_iou)

            rows.append({
                "variant": variant["name"],
                "benchmark": "temporal",
                "sequence": img_path.stem,
                "frame": fi,
                "difficulty": difficulty,
                "ground_truth": len(gt),
                "predictions": len(preds),
                "localized": m["localized"],
                "correct_id": m["correct_id"],
                "wrong_id": m["wrong_id"],
                "false_positive": m["fp"],
                "false_negative": m["fn"],
                "localization_precision": m["localized"] / max(len(preds), 1),
                "localization_recall": m["localized"] / max(len(gt), 1),
                "id_precision": m["correct_id"] / max(len(preds), 1),
                "id_recall": m["correct_id"] / max(len(gt), 1),
                "mean_iou": m["mean_iou"],
                "corner_rmse_px": m["mean_corner_rmse_px"],
                "center_error_px": m["mean_center_error_px"],
                "latency_ms": elapsed_ms,
                "aruco_ms": 1000.0 * stat_delta(s0, s1, "t_aruco", 0.0),
                "dedup_ms": 1000.0 * stat_delta(s0, s1, "t_dedup", 0.0),
                "track_ms": 1000.0 * stat_delta(s0, s1, "t_track", 0.0),
                "full_calls": stat_delta(s0, s1, "full", 0),
                "roi_calls": stat_delta(s0, s1, "roi", 0),
                "roi_retry": stat_delta(s0, s1, "roi_retry", 0),
                "roi_retry_ok": stat_delta(s0, s1, "roi_retry_ok", 0),
                "rescan": stat_delta(s0, s1, "rescan", 0),
                "raw_candidates": stat_delta(s0, s1, "raw", 0),
                "kept": stat_delta(s0, s1, "kept", 0),
                "removed": stat_delta(s0, s1, "removed", 0),
                "scene_texture": scene.get("ground_texture", ""),
                "altitude_m": scene.get("altitude_ref_m", None),
                "motion_blur_px": scene.get("motion_blur_len_px", None),
                "defocus_ksize": scene.get("defocus_blur_ksize", None),
                "vibration_px": scene.get("vibration_px", None),
                "visibility_mean": mean_or_zero(
                    [float(x.get("visible_fraction", 0.0)) for x in gt]
                ),
                "marker_side_mean_px": mean_or_zero(
                    [float(x.get("marker_side_px", 0.0)) for x in gt]
                ),
            })

    return rows, aggregate_variant(rows, variant["name"], "temporal")


def aggregate_variant(
    rows: List[Dict[str, Any]],
    variant: str,
    benchmark: str,
) -> Dict[str, Any]:
    if not rows:
        return {"variant": variant, "benchmark": benchmark, "frames": 0}

    gt = sum(int(r["ground_truth"]) for r in rows)
    pred = sum(int(r["predictions"]) for r in rows)
    loc = sum(int(r["localized"]) for r in rows)
    correct = sum(int(r["correct_id"]) for r in rows)
    wrong = sum(int(r["wrong_id"]) for r in rows)
    fp = sum(int(r["false_positive"]) for r in rows)
    fn = sum(int(r["false_negative"]) for r in rows)

    lat = [float(r["latency_ms"]) for r in rows]
    aruco = [float(r["aruco_ms"]) for r in rows]
    dedup = [float(r["dedup_ms"]) for r in rows]
    track = [float(r["track_ms"]) for r in rows]

    lp = loc / max(pred, 1)
    lr = loc / max(gt, 1)
    ip = correct / max(pred, 1)
    ir = correct / max(gt, 1)

    return {
        "variant": variant,
        "benchmark": benchmark,
        "frames": len(rows),
        "gt_markers": gt,
        "predictions": pred,
        "localized": loc,
        "correct_id": correct,
        "wrong_id": wrong,
        "false_positive": fp,
        "false_negative": fn,
        "localization_precision": lp,
        "localization_recall": lr,
        "localization_f1": f1_score(lp, lr),
        "id_precision": ip,
        "id_recall": ir,
        "id_f1": f1_score(ip, ir),
        "id_error_rate_among_localized": wrong / max(loc, 1),
        "mean_latency_ms": mean_or_zero(lat),
        "p50_latency_ms": pct(lat, 50),
        "p95_latency_ms": pct(lat, 95),
        "p99_latency_ms": pct(lat, 99),
        "max_latency_ms": max(lat),
        "fps_equivalent_mean": 1000.0 / max(mean_or_zero(lat), 1e-9),
        "frames_le_16_67ms_pct": 100.0 * sum(x <= 16.67 for x in lat) / len(lat),
        "frames_le_8_33ms_pct": 100.0 * sum(x <= 8.33 for x in lat) / len(lat),
        "mean_aruco_ms": mean_or_zero(aruco),
        "p95_aruco_ms": pct(aruco, 95),
        "mean_dedup_ms": mean_or_zero(dedup),
        "mean_track_ms": mean_or_zero(track),
        "full_calls": sum(int(r["full_calls"]) for r in rows),
        "roi_calls": sum(int(r["roi_calls"]) for r in rows),
        "roi_retry": sum(int(r["roi_retry"]) for r in rows),
        "roi_retry_ok": sum(int(r["roi_retry_ok"]) for r in rows),
        "rescan": sum(int(r["rescan"]) for r in rows),
        "raw_candidates": sum(int(r["raw_candidates"]) for r in rows),
        "kept": sum(int(r["kept"]) for r in rows),
        "removed": sum(int(r["removed"]) for r in rows),
    }


# ---------------------------------------------------------------------------
# Analyse par difficulté / taille / visibilité / texture
# ---------------------------------------------------------------------------

def stratified(rows: List[Dict[str, Any]], variant: str, benchmark: str) -> List[Dict[str, Any]]:
    if not rows:
        return []

    # Les métriques par taille utilisent la taille moyenne de la scène. Pour
    # une analyse encore plus fine, le JSON GT reste la source de vérité.
    def side_bin(v: float) -> str:
        if v < 14:
            return "<14 px"
        if v < 18:
            return "14-18 px"
        if v < 22:
            return "18-22 px"
        if v < 26:
            return "22-26 px"
        if v < 32:
            return "26-32 px"
        return ">=32 px"

    def vis_bin(v: float) -> str:
        if v < 0.4:
            return "<0.40"
        if v < 0.7:
            return "0.40-0.70"
        if v < 0.9:
            return "0.70-0.90"
        return ">=0.90"

    groups: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        keys = {
            "difficulty": str(r["difficulty"]),
            "side_bin": side_bin(float(r["marker_side_mean_px"])),
            "visibility_bin": vis_bin(float(r["visibility_mean"])),
            "texture": str(r["scene_texture"] or "unknown"),
        }
        for kind, value in keys.items():
            k = f"{kind}:{value}"
            groups.setdefault(k, []).append(r)

    out = []
    for key, rr in sorted(groups.items()):
        gt = sum(int(r["ground_truth"]) for r in rr)
        loc = sum(int(r["localized"]) for r in rr)
        correct = sum(int(r["correct_id"]) for r in rr)
        pred = sum(int(r["predictions"]) for r in rr)
        lat = [float(r["latency_ms"]) for r in rr]
        out.append({
            "variant": variant,
            "benchmark": benchmark,
            "group": key.split(":", 1)[0],
            "value": key.split(":", 1)[1],
            "frames": len(rr),
            "gt_markers": gt,
            "predictions": pred,
            "localization_recall": loc / max(gt, 1),
            "id_recall": correct / max(gt, 1),
            "id_precision": correct / max(pred, 1),
            "wrong_id": sum(int(r["wrong_id"]) for r in rr),
            "false_positive": sum(int(r["false_positive"]) for r in rr),
            "mean_latency_ms": mean_or_zero(lat),
            "p95_latency_ms": pct(lat, 95),
        })
    return out


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

def render_markdown(
    path: Path,
    system: Dict[str, Any],
    dataset_info: Dict[str, Any],
    summaries: List[Dict[str, Any]],
) -> None:
    lines = []
    lines.append("# Benchmark ArUco\n")
    lines.append(f"- Date : `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Python : `{system['python']}`")
    lines.append(f"- OpenCV : `{system['opencv']}`")
    lines.append(f"- NumPy : `{system['numpy']}`")
    lines.append(f"- CPU : `{system['cpu']}`")
    lines.append(f"- Threads OpenCV demandés : `{system['threads']}`")
    lines.append(f"- Dataset : `{dataset_info['root']}`")
    lines.append(f"- Images utilisées : `{dataset_info['images']}`")
    lines.append("")

    lines.append("## Résultats principaux\n")
    headers = [
        "variant", "benchmark", "id_recall", "id_precision", "id_f1",
        "wrong_id", "fp", "fn", "mean_ms", "p95_ms", "p99_ms",
        "fps_eq", "aruco_ms", "dedup_ms", "track_ms", "full", "roi",
        "retry_ok", "rescan",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for s in summaries:
        vals = [
            s["variant"],
            s["benchmark"],
            f"{100*s['id_recall']:.2f}%",
            f"{100*s['id_precision']:.2f}%",
            f"{100*s['id_f1']:.2f}%",
            s["wrong_id"], s["false_positive"], s["false_negative"],
            f"{s['mean_latency_ms']:.3f}",
            f"{s['p95_latency_ms']:.3f}",
            f"{s['p99_latency_ms']:.3f}",
            f"{s['fps_equivalent_mean']:.1f}",
            f"{s['mean_aruco_ms']:.3f}",
            f"{s['mean_dedup_ms']:.3f}",
            f"{s['mean_track_ms']:.3f}",
            s["full_calls"], s["roi_calls"],
            s["roi_retry_ok"], s["rescan"],
        ]
        lines.append("| " + " | ".join(map(str, vals)) + " |")

    lines.append("")
    lines.append("## Lecture des métriques\n")
    lines.append(
        "- `id_recall` : proportion des vrais marqueurs retrouvés avec le bon ID."
    )
    lines.append(
        "- `id_precision` : proportion des prédictions correspondant au bon ID."
    )
    lines.append(
        "- `localization_recall` : marqueur correctement localisé même si l'ID est faux."
    )
    lines.append(
        "- `wrong_id` : bonne localisation mais mauvais ID valide."
    )
    lines.append(
        "- `mean/p95/p99` : latence de `det.detect()` uniquement, hors lecture PNG."
    )
    lines.append(
        "- `aruco/dedup/track` : détail interne chronométré par la v2.1."
    )
    lines.append(
        "- `fps_eq` : 1000 / latence moyenne ; ce n'est pas un FPS caméra garanti."
    )
    lines.append("")

    lines.append("## Attention benchmark temporel\n")
    lines.append(
        "Le benchmark temporel réutilise les scènes générées mais applique des petites "
        "transformations géométriques déterministes pour solliciter le tracking et les ROI. "
        "Il est utile pour comparer les stratégies ROI, mais ne remplace pas une vraie "
        "séquence vidéo issue du vol."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def save_plots(outdir: Path, summary_rows: List[Dict[str, Any]], detail_rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[INFO] matplotlib indisponible : pas de graphiques.")
        return

    # 1) Latence moyenne / p95
    labels = [r["variant"] + "\n" + r["benchmark"] for r in summary_rows]
    meanv = [float(r["mean_latency_ms"]) for r in summary_rows]
    p95v = [float(r["p95_latency_ms"]) for r in summary_rows]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, meanv, width=0.36, label="moyenne")
    ax.bar(x + 0.18, p95v, width=0.36, label="p95")
    ax.set_ylabel("ms / image")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Latence moyenne / P95")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "latency.png", dpi=150)
    plt.close(fig)

    # 2) Recall ID
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    rec = [100.0 * float(r["id_recall"]) for r in summary_rows]
    ax.bar(x, rec)
    ax.set_ylabel("rappel ID (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Rappel ID")
    fig.tight_layout()
    fig.savefig(outdir / "recall.png", dpi=150)
    plt.close(fig)

    # 3) Rappel par taille de marqueur pour le benchmark static
    small = [r for r in detail_rows if r.get("benchmark") == "static"]
    if small:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        # Agrégation grossière directement depuis les lignes image.
        bins = ["<14 px", "14-18 px", "18-22 px", "22-26 px", "26-32 px", ">=32 px"]
        for v in sorted({r["variant"] for r in small}):
            pts = []
            for b in bins:
                rr = []
                for r in small:
                    if r["variant"] != v:
                        continue
                    side = float(r["marker_side_mean_px"])
                    if b == "<14 px" and side < 14:
                        rr.append(r)
                    elif b == "14-18 px" and 14 <= side < 18:
                        rr.append(r)
                    elif b == "18-22 px" and 18 <= side < 22:
                        rr.append(r)
                    elif b == "22-26 px" and 22 <= side < 26:
                        rr.append(r)
                    elif b == "26-32 px" and 26 <= side < 32:
                        rr.append(r)
                    elif b == ">=32 px" and side >= 32:
                        rr.append(r)
                gt = sum(int(r["ground_truth"]) for r in rr)
                ok = sum(int(r["correct_id"]) for r in rr)
                pts.append(100.0 * ok / max(gt, 1) if rr else np.nan)
            ax.plot(bins, pts, marker="o", label=v)

        ax.set_ylim(0, 100)
        ax.set_ylabel("rappel ID (%)")
        ax.set_xlabel("taille moyenne du marqueur")
        ax.set_title("Rappel ID selon la taille du marqueur")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "recall_by_marker_size.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark complet aruco_dedup_v2 + dataset réaliste"
    )

    here = Path(__file__).resolve().parent

    # Priorité aux fichiers fournis avec cette version de l'audit :
    #   aruco_dedup_v2(2).py = détecteur v2.1 actuel
    #   generate_realistic_dataset(1).py = générateur actuel
    preferred_detector = here / "aruco_dedup_v2(2).py"
    preferred_generator = here / "generate_realistic_dataset(1).py"

    default_detector = preferred_detector if preferred_detector.exists() else find_default_file("aruco_dedup_v2*.py")
    default_generator = preferred_generator if preferred_generator.exists() else find_default_file("generate_realistic_dataset*.py")

    p.add_argument("--detector", default=str(default_detector) if default_detector else "aruco_dedup_v2.py")
    p.add_argument("--generator", default=str(default_generator) if default_generator else "generate_realistic_dataset.py")
    p.add_argument("--dataset", default="benchmark_dataset")
    p.add_argument("--out", default="benchmark_results")

    p.add_argument("--generate", action="store_true",
                   help="force la génération si le dataset n'existe pas")
    p.add_argument("--regenerate", action="store_true",
                   help="supprime/recrée le dataset avant benchmark")
    p.add_argument("--scenes", type=int, default=200)
    p.add_argument("--markers-per-scene", default="1-3")
    p.add_argument("--marker-ids", default="0-40")
    p.add_argument("--difficulty", choices=["normal", "hard", "both"], default="normal")
    p.add_argument("--marker-size-m", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--presets", default="speed,balanced,quality")
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--unique-ids", action="store_true")
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--roi-retry-scale", type=float, default=1.7)
    p.add_argument("--track-max-miss", type=int, default=3)
    p.add_argument("--threads", type=int, default=3)

    p.add_argument("--include-no-roi", action="store_true")
    p.add_argument("--sweep", action="store_true",
                   help="teste roi_min_side et full_scan_every en plus des presets")
    p.add_argument("--sweep-threads", action="store_true",
                   help="teste OpenCV 1/2/3/4 threads en plus")

    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max-images", type=int, default=0,
                   help="0 = toutes les images du dataset")
    p.add_argument("--match-iou", type=float, default=0.20)

    p.add_argument("--no-temporal", action="store_true")
    p.add_argument("--temporal-scenes", type=int, default=60)
    p.add_argument("--temporal-frames", type=int, default=8)
    p.add_argument("--temporal-fps", type=float, default=30.0)

    p.add_argument("--progress", type=int, default=25,
                   help="afficher la progression toutes les N images, 0 = silencieux")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--save-failures", action="store_true",
                   help="sauvegarde quelques images présentant erreurs / latences élevées")
    p.add_argument("--max-failure-images", type=int, default=30)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    detector_path = Path(args.detector).resolve()
    generator_path = Path(args.generator).resolve()
    dataset_root = Path(args.dataset).resolve()
    outdir = Path(args.out).resolve()

    if not detector_path.exists():
        raise SystemExit(f"Détecteur introuvable : {detector_path}")
    if not generator_path.exists():
        raise SystemExit(f"Générateur introuvable : {generator_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Génération/recyclage dataset
    # ----------------------------------------------------------------------
    manifest = dataset_root / "benchmark_dataset_manifest.json"
    requested_manifest = {
        "generator": str(generator_path),
        "generator_sha256": sha256_file(generator_path),
        "scenes": args.scenes,
        "markers_per_scene": args.markers_per_scene,
        "marker_ids": args.marker_ids,
        "difficulty": args.difficulty,
        "marker_size_m": args.marker_size_m,
        "seed": args.seed,
    }

    need_generate = args.regenerate or not manifest.exists()
    if not need_generate and args.generate:
        try:
            old = json.loads(manifest.read_text(encoding="utf-8"))
            need_generate = old != requested_manifest
        except Exception:
            need_generate = True

    if need_generate:
        import shutil
        if dataset_root.exists() and args.regenerate:
            shutil.rmtree(dataset_root)
        run_generator(
            generator=generator_path,
            dataset_root=dataset_root,
            n_scenes=args.scenes,
            markers_per_scene=args.markers_per_scene,
            marker_ids=args.marker_ids,
            difficulty=args.difficulty,
            seed=args.seed,
            marker_size_m=args.marker_size_m,
        )
        json_dump(manifest, requested_manifest)
    else:
        print(f"[DATASET] réutilisation : {dataset_root}")

    items = dataset_files(dataset_root, args.max_images)
    if not items:
        raise SystemExit("Aucune paire PNG/JSON trouvée dans le dataset.")

    # ----------------------------------------------------------------------
    # Charger le détecteur exact fourni
    # ----------------------------------------------------------------------
    mod = load_module_from_path(detector_path, "aruco_benchmark_detector")
    detector_version = getattr(mod, "__version__", "unknown")

    system = {
        "python": sys.version.replace("\n", " "),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.uname().processor,
        "threads": cv2.getNumThreads(),
        "detector_version": detector_version,
        "detector_path": str(detector_path),
        "detector_sha256": sha256_file(detector_path),
    }

    dataset_info = {
        "root": str(dataset_root),
        "images": len(items),
        "difficulties": sorted(set(x[2] for x in items)),
        "generator": str(generator_path),
        "generator_sha256": sha256_file(generator_path),
    }

    # Résultats
    variants = build_variants(args)
    summary_rows = []
    detail_rows = []
    strat_rows = []

    print("\n" + "=" * 78)
    print("BENCHMARK ARUCO")
    print("=" * 78)
    print(f"Détecteur : {detector_path}")
    print(f"Version   : {detector_version}")
    print(f"Dataset   : {dataset_root}")
    print(f"Images    : {len(items)}")
    print(f"Variantes : {len(variants)}")
    print("=" * 78)

    for v in variants:
        rows, summary = benchmark_static(mod, v, items, args)
        detail_rows.extend(rows)
        summary_rows.append(summary)
        strat_rows.extend(stratified(rows, v["name"], "static"))

        if not args.no_temporal:
            rows_t, summary_t = benchmark_temporal(mod, v, items, args)
            detail_rows.extend(rows_t)
            summary_rows.append(summary_t)
            strat_rows.extend(stratified(rows_t, v["name"], "temporal"))

    # ----------------------------------------------------------------------
    # Sorties
    # ----------------------------------------------------------------------
    write_csv(outdir / "per_image.csv", detail_rows)
    write_csv(outdir / "summary.csv", summary_rows)
    write_csv(outdir / "stratified.csv", strat_rows)

    report_json = {
        "system": system,
        "dataset": dataset_info,
        "args": vars(args),
        "summaries": summary_rows,
        "stratified": strat_rows,
    }
    json_dump(outdir / "report.json", report_json)

    render_markdown(
        outdir / "report.md",
        system=system,
        dataset_info=dataset_info,
        summaries=summary_rows,
    )

    if args.plots:
        save_plots(outdir, summary_rows, detail_rows)

    # ----------------------------------------------------------------------
    # Affichage console : tableau principal
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RÉSULTATS")
    print("=" * 78)
    print(
        f"{'VARIANTE':28s} {'MODE':9s} "
        f"{'RCL':>7s} {'PREC':>7s} {'F1':>7s} "
        f"{'ms':>8s} {'p95':>8s} {'FPS':>7s} "
        f"{'ARUCO':>8s} {'ROI':>6s} {'FULL':>6s}"
    )
    print("-" * 78)

    for s in summary_rows:
        print(
            f"{s['variant']:28.28s} "
            f"{s['benchmark']:9.9s} "
            f"{100*s['id_recall']:6.2f}% "
            f"{100*s['id_precision']:6.2f}% "
            f"{100*s['id_f1']:6.2f}% "
            f"{s['mean_latency_ms']:8.3f} "
            f"{s['p95_latency_ms']:8.3f} "
            f"{s['fps_equivalent_mean']:7.1f} "
            f"{s['mean_aruco_ms']:8.3f} "
            f"{s['roi_calls']:6d} "
            f"{s['full_calls']:6d}"
        )

    print("-" * 78)
    print(f"CSV détail   : {outdir / 'per_image.csv'}")
    print(f"CSV résumé   : {outdir / 'summary.csv'}")
    print(f"CSV analyse  : {outdir / 'stratified.csv'}")
    print(f"Rapport MD   : {outdir / 'report.md'}")
    print(f"Rapport JSON : {outdir / 'report.json'}")
    if args.plots:
        print(f"Graphiques   : {outdir}")

    # Optionnel : quelques pires latences / erreurs en console.
    static_rows = [r for r in detail_rows if r["benchmark"] == "static"]
    if static_rows:
        worst = sorted(static_rows, key=lambda r: float(r["latency_ms"]), reverse=True)[:10]
        print("\nTop 10 latences statiques :")
        for r in worst:
            print(
                f"  {r['variant']:26s} {r['image']:18s} "
                f"{float(r['latency_ms']):8.3f} ms | "
                f"gt={r['ground_truth']} pred={r['predictions']} "
                f"correct={r['correct_id']} fp={r['false_positive']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
