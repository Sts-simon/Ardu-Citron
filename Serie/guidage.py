#!/usr/bin/env python3
"""
guidage.py - bibliotheque + outil en ligne de commande pour toute la serie "types de guidage".

Un episode = UN fichier (ex. ep01.py) qui contient :
    SCRIPT    le texte de la voix off, avec des marqueurs [[cle]] devant les mots qui declenchent l'image
    RELATIVE  (optionnel) cues deduites d'une autre : {"hook_out": "hook2+0.9"}
    Episode   la scene Manim (classe derivee de BlueprintScene), ecrite avec self.at("cle")

Commandes (a lancer depuis le dossier des fichiers) :
    python3 guidage.py ep01.py script                       texte a lire (sans marqueurs)
    python3 guidage.py ep01.py gaps                         ecarts minimaux mesures sur l'animation
    python3 guidage.py ep01.py draft                        cues estimees SANS voix (brouillon)
    python3 guidage.py ep01.py transcribe --voice voix.wav  WhisperX seul : build/words.json + build/ep01_mots.txt
                                                            (utile avant d'avoir ecrit la scene)
    python3 guidage.py ep01.py sync --voice voix.wav        WhisperX + cues recalees sur la voix
    python3 guidage.py ep01.py build [--voice voix.wav] [--preview]
                                                            synchro + rendu + bruitages + mixage -> build/*.mp4
Options de build : --skip-whisper (reutilise build/words.json) --skip-render --model small.en
                   --device cpu|cuda --compute-type int8|float16

Tout ce qui est genere va dans build/ a cote du fichier d'episode.
"""
import argparse
import difflib
import glob
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import wave
from collections import defaultdict

import numpy as np
from manim import *  # noqa: F401,F403
from manim import config

# =============================================================================================
# 1. CHARTE BLUEPRINT
# =============================================================================================
BG = "#0b2a5b"      # fond
INK = "#f4f8ff"     # structure
CYAN = "#8fe3ff"    # information active
AMBER = "#ffcf5a"   # conclusion (une seule fois)
GRIDC = "#5b83bd"
FONT = "DejaVu Sans Mono"
EQ_X0, ROW_Y0, ROW_DY = -3.8, 4.9, 0.44      # colonne de mots-cles en haut a gauche

PREVIEW = os.environ.get("PREVIEW") == "1"
MARGIN = 0.05          # marge ajoutee a chaque ecart minimal
MEASURE_FPS = 15       # mesure au pire cas (15 fps) : valable aussi pour 30 fps


def setup_config():
    pw = 540 if PREVIEW else 1080
    config.pixel_width = pw
    config.pixel_height = pw * 16 // 9
    config.frame_width = 9.0
    config.frame_height = 16.0
    config.frame_rate = 15 if PREVIEW else 30
    config.background_color = BG


setup_config()


# =============================================================================================
# 2. OUTILS GRAPHIQUES
# =============================================================================================
def TX(s, size=26, color=INK):
    return Text(s, font=FONT, font_size=size, color=color)


def block(label, w, h, size=22, color=INK):
    r = RoundedRectangle(corner_radius=0.09, width=w, height=h, stroke_color=color,
                         stroke_width=1.8, fill_color=BG, fill_opacity=1)
    lm = TX(label, size, color)
    if lm.width > w - 0.22 or lm.height > h - 0.12:
        lm.scale(min((w - 0.22) / lm.width, (h - 0.12) / lm.height))
    lm.move_to(r.get_center())
    return VGroup(r, lm)


def arr(p, q):
    return Arrow(p, q, buff=0, stroke_width=2.4, tip_length=0.15, color=INK,
                 max_tip_length_to_length_ratio=0.6)


def vec(p, d, length, color=CYAN):
    d = np.array(d, float)
    d = d / np.linalg.norm(d)
    return Arrow(p, p + d * length, buff=0, color=color, stroke_width=2.6, tip_length=0.16,
                 max_tip_length_to_length_ratio=0.5)


def fit_right(m, xmax=3.45):
    """Reduit un texte qui depasserait la zone utile (1 unite libre a droite pour l'interface Reels)."""
    if m.get_right()[0] > xmax:
        m.scale((xmax - m.get_left()[0]) / m.width, about_point=m.get_left())
    return m


def make_hook(line1, line2):
    """Accroche en deux temps : (groupe titre+filet, second mot en cyan)."""
    h = TX(line1, 44, INK).move_to([0, 5.2, 0])
    rule = Line([-3.0, 4.72, 0], [3.0, 4.72, 0], color=CYAN, stroke_width=2.0)
    return VGroup(h, rule).set_z_index(10), TX(line2, 44, CYAN).move_to([0, 4.2, 0]).set_z_index(10)


def ground_cote(xa, xb, gy, label="Δ"):
    """Cote horizontale (double fleche cyan + traits de rappel) posee sur le sol d'ordonnee gy."""
    lo, hi = sorted((xa, xb))
    yc = gy + 0.85
    ex = VGroup(*[DashedLine([x, gy + 0.12, 0], [x, yc + 0.2, 0], dash_length=0.08,
                             stroke_width=1.2, color=CYAN) for x in (xa, xb)])
    ar = DoubleArrow([lo, yc, 0], [hi, yc, 0], buff=0, stroke_width=2.4, tip_length=0.16,
                     color=CYAN, max_tip_length_to_length_ratio=0.45)
    lb = TX(label, 36, CYAN).move_to([(lo + hi) / 2, yc + 0.38, 0])
    return VGroup(ex, ar, lb).set_z_index(5)


def target_at(x, gy):
    """Reticule pose sur le sol."""
    return VGroup(Circle(radius=0.26, color=INK, stroke_width=2.5),
                  *[Line(d * 0.18, d * 0.48, color=INK, stroke_width=2.2)
                    for d in (UP, DOWN, LEFT, RIGHT)]).move_to([x, gy + 0.5, 0]).set_z_index(4)


def _outline(half, closed_tail=True):
    pts = half + [(x, -y) for x, y in reversed(half[1:-1])]
    return Polygon(*[[x, y, 0] for x, y in pts], stroke_color=INK, stroke_width=2.4,
                   fill_color=BG, fill_opacity=1)


def aircraft_shape():
    body = [(0.70, 0.00), (0.42, 0.07), (0.20, 0.16), (0.00, 0.16), (-0.12, 0.08), (-0.40, 0.08),
            (-0.58, 0.40), (-0.72, 0.40), (-0.68, 0.06), (-0.74, -0.02), (-0.56, -0.11), (0.38, -0.11)]
    poly = Polygon(*[[x, y, 0] for x, y in body], stroke_color=INK, stroke_width=2.6,
                   fill_color=BG, fill_opacity=1)
    wing = Polygon([0.18, -0.02, 0], [-0.22, -0.02, 0], [-0.40, -0.17, 0], [-0.14, -0.17, 0],
                   fill_opacity=0)
    det = VGroup(wing, Line([0.30, 0.09, 0], [0.02, 0.09, 0])).set_stroke(INK, 1.4, opacity=0.75)
    return VGroup(poly, det).scale(1.15)


def bomb_shape():
    half = [(0.32, 0.0), (0.26, 0.055), (0.10, 0.085), (-0.16, 0.085), (-0.26, 0.15),
            (-0.34, 0.15), (-0.30, 0.04), (-0.34, 0.0)]
    det = VGroup(Line([0.06, -0.085, 0], [0.06, 0.085, 0]),
                 Line([-0.16, -0.085, 0], [-0.16, 0.085, 0])).set_stroke(INK, 1.3, opacity=0.75)
    return VGroup(_outline(half), det)


def rocket_shape():
    half = [(0.42, 0.0), (0.36, 0.045), (0.22, 0.06), (-0.28, 0.06), (-0.36, 0.13),
            (-0.44, 0.13), (-0.40, 0.03), (-0.44, 0.0)]
    det = VGroup(Line([0.14, -0.06, 0], [0.14, 0.06, 0]),
                 Line([-0.1, -0.06, 0], [-0.1, 0.06, 0])).set_stroke(INK, 1.3, opacity=0.75)
    return VGroup(_outline(half), det)


def missile_shape(scale=1.4):
    """Missile du premier film (vertical, pointe vers le haut)."""
    R = [(0, 1.0), (0.04, 0.93), (0.08, 0.82), (0.11, 0.68), (0.125, 0.52), (0.125, -0.04),
         (0.56, -0.40), (0.56, -0.50), (0.125, -0.50), (0.125, -0.80), (0.33, -0.93),
         (0.33, -1.0), (0.125, -0.97), (0.10, -1.0)]
    pts = R + [(-x, y) for x, y in reversed(R[1:])]
    poly = Polygon(*[[x, y, 0] for x, y in pts], stroke_color=INK, stroke_width=2.6,
                   fill_color=BG, fill_opacity=1)
    det = VGroup(Line([-0.11, 0.68, 0], [0.11, 0.68, 0]), Line([-0.125, 0.30, 0], [0.125, 0.30, 0]),
                 Line([-0.125, -0.04, 0], [0.125, -0.04, 0]), Line([-0.125, -0.50, 0], [0.125, -0.50, 0]),
                 Line([0, 0.68, 0], [0, -0.80, 0])).set_stroke(INK, 1.4, opacity=0.75)
    return VGroup(poly, det).scale(scale)


# =============================================================================================
# 3. PHYSIQUE ILLUSTRATIVE + TRAJECTOIRES / MOUVEMENTS PILOTES PAR LE TEMPS
# =============================================================================================
def simulate(p0, v0, g=3.05, ground=-3.65, k=0.0, wind=(0.0, 0.0), gust=None, dt=0.008):
    """Chute avec gravite g, trainee quadratique k, vent et rafale (instant, vecteur). Renvoie un tableau (N,3)."""
    p = np.array(p0, float)[:2]
    v = np.array(v0, float)[:2]
    pts, t = [p.copy()], 0.0
    while len(pts) < 6000:
        w = np.array(wind, float)
        if gust and t >= gust[0]:
            w = w + np.array(gust[1], float)
        rel = v - w
        a = np.array([0.0, -g]) - k * np.linalg.norm(rel) * rel
        v = v + a * dt
        pn = p + v * dt
        t += dt
        if pn[1] <= ground:
            u = (p[1] - ground) / (p[1] - pn[1])
            pts.append(p + (pn - p) * u)
            break
        p = pn
        pts.append(p.copy())
    pts = np.array(pts)
    return np.column_stack([pts, np.zeros(len(pts))])


def shift_path(P, x, y):
    """Meme trajectoire, depart decale en (x, y)."""
    return P - P[0] + np.array([x, y, 0.0])


class Flight:
    """Trajectoire parcourue entre t0 et t0+dur (temps de la scene) ; sa trace grandit avec elle."""

    def __init__(self, scene, pts, t0, dur, color=CYAN, width=2.4):
        self.s, self.pts, self.t0, self.dur = scene, pts, t0, max(dur, 1e-3)
        self.done = False
        self.trail = VMobject(stroke_color=color, stroke_width=width).set_z_index(3)
        self.trail.set_points_as_corners([pts[0], pts[0] + 0.001 * RIGHT])

        def up(m, dt):
            if self.done:
                return
            n = int(self.idx()) + 2
            m.set_points_as_corners(self.pts[:min(n, len(self.pts))])
            if self.a() >= 1.0:
                self.done = True
        self.trail.add_updater(up)

    def a(self):
        return min(max((self.s.time - self.t0) / self.dur, 0.0), 1.0)

    def idx(self):
        return self.a() * (len(self.pts) - 1)

    def pos(self):
        f = self.idx()
        i = int(f)
        if i >= len(self.pts) - 1:
            return self.pts[-1].copy()
        u = f - i
        return self.pts[i] * (1 - u) + self.pts[i + 1] * u

    def angle(self):
        n = len(self.pts)
        i = int(self.idx())
        d = self.pts[min(i + 2, n - 1)] - self.pts[max(i - 1, 0)]
        return float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-9 else 0.0


class Track:
    """Abscisse d'un objet (ex. l'avion) pilotee par le temps : glissades successives."""

    def __init__(self, scene, x0):
        self.s, self.x0, self.segs = scene, x0, []

    def val(self, t=None):
        t = self.s.time if t is None else t
        x = self.x0
        for (t0, t1, xa, xb, sm) in self.segs:
            if t >= t1:
                x = xb
            elif t >= t0:
                u = (t - t0) / (t1 - t0)
                if sm:
                    u = u * u * (3 - 2 * u)
                return xa + (xb - xa) * u
            else:
                break
        return x

    def glide(self, xb, t0, t1, smooth=True, sound=True):
        """t0, t1 : nombre, 'cle' ou ('cle', decalage)."""
        a, b = self.s.T(t0), self.s.T(t1)
        self.segs.append((a, b, self.val(a), xb, smooth))
        if sound:
            self.s.sfx("whoosh", t=a, dur=max(b - a, 0.3), f0=200, f1=900, gain=0.3)


# =============================================================================================
# 4. SCENE DE BASE : cues, mesure des ecarts, evenements sonores
# =============================================================================================
class BlueprintScene(Scene):
    """
    Ecris la timeline avec  self.at("cle")  (ou self.at("cle", +decalage)) avant chaque action.
    - Rendu normal : at() attend l'instant de la cle lu dans build/cues.json.
    - Mode mesure (CLI gaps/draft/sync/build) : la scene est executee SANS rendu ; chaque at() releve
      le temps ecoule depuis le precedent, ce qui donne les ecarts minimaux entre cues.
    Le son est deduit de ce qui se passe : FadeIn/Create -> tic, Flash -> ping, flight()/glide() -> souffle,
    plus des evenements explicites : self.sfx("clack"|"thud"|"crackle"|"rise"|"impact"|"lock"|"whoosh"|"tick").
    Dans self.play(...), sfx=False coupe le son automatique, sfx="tick" force un type.
    """

    def __init__(self, *args, measure=False, **kwargs):
        self.measure = measure
        self._vt = 0.0
        self._last = None
        self.order, self.edges, self.needs = [], [], []
        self.sfx_events, self.late = [], []
        self._cues = None
        if not measure:
            super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------- temps
    @property
    def time(self):
        return self._vt if self.measure else super().time

    def cue(self, key):
        if self.measure:
            return 0.0
        if self._cues is None:
            path = os.environ.get("GUIDAGE_CUES", os.path.join(os.getcwd(), "build", "cues.json"))
            if not os.path.exists(path):
                raise SystemExit(f"cues introuvables ({path}) : lance d'abord 'draft' ou 'sync', ou 'build'.")
            self._cues = json.load(open(path, encoding="utf-8"))["C"]
        if key not in self._cues:
            raise SystemExit(f"cue inconnue dans cues.json : {key}")
        return self._cues[key]

    def T(self, spec):
        if isinstance(spec, str):
            return self.cue(spec)
        if isinstance(spec, tuple):
            return self.cue(spec[0]) + spec[1]
        return float(spec)

    def at(self, key, off=0.0):
        if self.measure:
            if key not in self.order:
                self.order.append(key)
            if self._last is not None and self._last[0] != key:
                k0, off0, vt0 = self._last
                self.edges.append((key, k0, off0 + (self._vt - vt0) - off))
            self._last = (key, off, self._vt)
            return
        t = self.cue(key) + off
        d = t - self.time
        frame = 1.0 / config.frame_rate
        if d > frame:
            self.wait(d)
        elif d < -frame:
            self.late.append((key, round(t, 2), round(-d, 3)))

    def need(self, key, after, gap):
        """Contrainte supplementaire : cue[key] >= cue[after] + gap (ex. duree minimale d'un vol)."""
        if self.measure:
            self.needs.append((key, after, gap))

    # ---------------------------------------------------------------- primitives surchargees
    def play(self, *args, **kwargs):
        sfx = kwargs.pop("sfx", True)
        if self.measure:
            rt = kwargs.get("run_time")
            if rt is None:
                rts = [a.run_time for a in args if hasattr(a, "run_time")]
                rt = max(rts) if rts else 1.0
            self._vt += math.ceil(rt * MEASURE_FPS - 1e-6) / MEASURE_FPS
            return
        self._auto_sfx(args, sfx)
        super().play(*args, **kwargs)

    def wait(self, duration=1.0, *args, **kwargs):
        if self.measure:
            self._vt += math.ceil(duration * MEASURE_FPS - 1e-6) / MEASURE_FPS
            return
        super().wait(duration, *args, **kwargs)

    def add(self, *mobjects):
        if self.measure:
            return self
        return super().add(*mobjects)

    def remove(self, *mobjects):
        if self.measure:
            return self
        return super().remove(*mobjects)

    # ---------------------------------------------------------------- son
    def _auto_sfx(self, args, sfx):
        if sfx is False:
            return
        t = self.time
        if isinstance(sfx, str):
            self.sfx(sfx, t=t)
            return
        names = {type(a).__name__ for a in args}
        if names & {"FadeIn", "Create", "Write", "DrawBorderThenFill", "GrowFromCenter"}:
            self.sfx("tick", t=t)
        if "Flash" in names:
            self.sfx("ping", t=t)
        elif "Indicate" in names and not (names & {"FadeIn", "Create"}):
            self.sfx("tick", t=t, gain=0.8)

    def sfx(self, kind, t=None, **kw):
        if self.measure:
            return
        self.sfx_events.append(dict(t=round(self.time if t is None else self.T(t), 3), kind=kind, **kw))

    def tear_down(self):
        path = os.environ.get("GUIDAGE_SFX_EVENTS")
        if path:
            json.dump({"end": self.time, "events": sorted(self.sfx_events, key=lambda e: e["t"])},
                      open(path, "w"), indent=1)
        if self.late:
            print("SYNC WARNINGS - cue / instant vise / retard (s) :")
            for key, t, d in self.late:
                print(f"  {key:<14}{t:7.2f}s  +{d:.3f}s")
        else:
            print("SYNC CHECK: aucune cue en retard de plus d'une image.")

    # ---------------------------------------------------------------- confort
    def flight(self, pts, start, end=None, dur=None, clack=False, thud=0.9, f0=250, f1=1400,
               gain=0.6, color=CYAN, width=2.4):
        """Lance une trajectoire. start/end : cles (ou ('cle', decalage)) ; ou dur en secondes."""
        t0 = self.T(start)
        d = dur if dur is not None else self.T(end) - t0
        f = Flight(self, pts, t0, d, color, width)
        self.fl = f
        self.add(f.trail)
        self.sfx("whoosh", t=t0, dur=max(d, 0.3), f0=f0, f1=f1, gain=gain)
        if thud:
            self.sfx("thud", t=t0 + d, gain=thud)
        if clack:
            self.sfx("clack", t=t0)
        return f

    def show(self, *mobs, rt=0.35):
        self.play(*[FadeIn(m) for m in mobs], run_time=rt)

    def rowt(self, s, i, size=24, color=INK):
        return TX(s, size, color).move_to([EQ_X0, ROW_Y0 - ROW_DY * i, 0], aligned_edge=LEFT).set_z_index(9)

    def ping(self, pos):
        return Flash(pos, color=CYAN, flash_radius=0.4, line_length=0.14, num_lines=10,
                     line_stroke_width=2.2, run_time=0.5)

    def background(self, end_key="end"):
        """Grille blueprint + cadre + balayage cyan (la video boucle : periode = duree/8)."""
        grid = VGroup()
        for i in range(-9, 10):
            maj = i % 4 == 0
            grid.add(Line([i * 0.5, -8, 0], [i * 0.5, 8, 0], color=GRIDC,
                          stroke_width=1.0 if maj else 0.55, stroke_opacity=0.22 if maj else 0.075))
        for j in range(-16, 17):
            maj = j % 4 == 0
            grid.add(Line([-4.5, j * 0.5, 0], [4.5, j * 0.5, 0], color=GRIDC,
                          stroke_width=1.0 if maj else 0.55, stroke_opacity=0.22 if maj else 0.075))
        border = Rectangle(width=8.5, height=15.5, color=INK, stroke_width=1.2, stroke_opacity=0.42)
        grid.set_z_index(0)
        border.set_z_index(0)
        total = max(self.cue(end_key), 8.0)
        sweep = Line([-4.25, 0, 0], [4.25, 0, 0], color=CYAN, stroke_width=1.6,
                     stroke_opacity=0.10).set_z_index(1)
        sweep.add_updater(lambda m, dt: m.move_to([0, 8 - 16 * ((self.time / (total / 8)) % 1), 0]))
        return grid, border, sweep


# =============================================================================================
# 5. SCRIPT MARQUE, TRANSCRIPTION, ALIGNEMENT, CALCUL DES CUES
# =============================================================================================
def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def parse_script(text):
    """-> dict(words, raw, markers[(cle, idx, 'start'|'end', decalage)], pauses{idx: s})."""
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    words, raw, markers, pauses, pending = [], [], [], defaultdict(float), []
    for m in re.finditer(r"\[\[[^\]]+\]\]|\{[^}]*\}|[^\s\[\]{}]+", text):
        tok = m.group(0)
        if tok.startswith("[["):
            mm = re.fullmatch(r"\[\[(\w+)(?:@(end))?([+-]\d+(?:\.\d+)?)?\]\]", tok)
            if not mm:
                sys.exit(f"Marqueur invalide : {tok}")
            pending.append((mm.group(1), "end" if mm.group(2) else "start", float(mm.group(3) or 0.0)))
        elif tok.startswith("{"):
            pm = re.fullmatch(r"\{pause\s+([\d.]+)\s*s?\}", tok)
            if pm and words:
                pauses[len(words) - 1] += float(pm.group(1))
        else:
            for part in tok.split("-"):
                n = norm(part)
                if n:
                    for key, mode, off in pending:
                        markers.append((key, len(words), mode, off))
                    pending = []
                    words.append(n)
                    raw.append(part)
    for key, mode, off in pending:
        markers.append((key, len(words) - 1, mode, off))
    return dict(words=words, raw=raw, markers=markers, pauses=pauses)


def reading_text(text):
    t = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    t = re.sub(r"\[\[[^\]]+\]\]", "", t)
    t = re.sub(r"\{pause\s+([\d.]+)\s*s?\}", r"(pause \1 s)", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def draft_words(ps):
    """Transcription fictive (debit ~2,4 mots/s + pauses) pour caler l'animation avant d'avoir la voix."""
    t, out = 0.5, []
    for i, w in enumerate(ps["words"]):
        d = 0.10 + 0.048 * len(w)
        out.append({"word": w, "start": t, "end": t + d})
        t += d + 0.05
        r = ps["raw"][i]
        t += 0.4 if r.endswith((".", "!", "?", ":")) else 0.15 if r.endswith(",") else 0.0
        t += ps["pauses"].get(i, 0.0)
    return out


def load_words(path):
    ws = [w for w in json.load(open(path, encoding="utf-8")) if norm(w.get("word", ""))]
    for i, w in enumerate(ws):
        if w.get("start") is None or w.get("end") is None:
            prev = next((ws[j] for j in range(i - 1, -1, -1) if ws[j].get("end") is not None), None)
            nxt = next((ws[j] for j in range(i + 1, len(ws)) if ws[j].get("start") is not None), None)
            a = prev["end"] if prev else (nxt["start"] - 0.3 if nxt else 0.0)
            b = nxt["start"] if nxt else a + 0.3
            w["start"], w["end"] = a, max(a + 0.05, b)
    return ws


def merge_letters(ws):
    """'C' 'C' 'I' 'P' -> 'ccip' (WhisperX epelle souvent les sigles)."""
    out, i = [], 0
    while i < len(ws):
        j = i
        while j < len(ws) and len(norm(ws[j]["word"])) == 1 and norm(ws[j]["word"]) not in ("a", "i"):
            j += 1
        if j - i >= 2:
            out.append({"word": "".join(norm(w["word"]) for w in ws[i:j]),
                        "start": ws[i]["start"], "end": ws[j - 1]["end"]})
            i = j
        else:
            out.append(ws[i])
            i += 1
    return out


def match(script, asr):
    """Apparie les mots du script a ceux de la transcription : idx_script -> (idx_asr, 'exact'|'fuzzy')."""
    b = [norm(w["word"]) for w in asr]
    sm = difflib.SequenceMatcher(None, script, b, autojunk=False)
    mp = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            mp[blk.a + k] = (blk.b + k, "exact")
    last_b = -1
    for i in range(len(script)):
        if i in mp:
            last_b = mp[i][0]
            continue
        nxt_b = next((mp[j][0] for j in range(i + 1, len(script)) if j in mp), len(b))
        best, best_r = None, 0.0
        for jb in range(last_b + 1, nxt_b):
            r = difflib.SequenceMatcher(None, script[i], b[jb]).ratio()
            if r > best_r:
                best, best_r = jb, r
        if best is not None and best_r >= 0.72 and len(script[i]) >= 3:
            mp[i] = (best, "fuzzy")
            last_b = best
    return mp


def word_time(i, mp, asr, mode):
    if i in mp:
        w = asr[mp[i][0]]
        return (w["start"] if mode == "start" else w["end"]), mp[i][1]
    prev = next((j for j in range(i - 1, -1, -1) if j in mp), None)
    nxt = next((j for j in range(i + 1, 10 ** 6) if j in mp), None) if mp else None
    if prev is not None and nxt is not None:
        t0, t1 = asr[mp[prev][0]]["end"], asr[mp[nxt][0]]["start"]
        t = t0 + (t1 - t0) * (i - prev) / (nxt - prev)
    elif prev is not None:
        t = asr[mp[prev][0]]["end"] + 0.35 * (i - prev)
    elif nxt is not None:
        t = asr[mp[nxt][0]]["start"] - 0.35 * (nxt - i)
    else:
        sys.exit("Aucun mot du script retrouve dans la transcription.")
    return t, "interp"


def load_episode(path):
    path = os.path.abspath(path)
    sys.path.insert(0, os.path.dirname(path))
    if __name__ == "__main__":
        sys.modules.setdefault("guidage", sys.modules["__main__"])
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for need in ("SCRIPT", "Episode"):
        if not hasattr(mod, need):
            sys.exit(f"{path} doit definir {need}")
    return mod


def measure(ep):
    sc = ep.Episode(measure=True)
    sc.construct()
    return sc.order, sc.edges, sc.needs


def parse_relative(rel):
    out = {}
    for k, spec in rel.items():
        m = re.fullmatch(r"(\w+)\s*([+-])\s*(\d+(?:\.\d+)?)", spec.replace(" ", ""))
        if not m:
            sys.exit(f"RELATIVE invalide pour {k} : {spec!r} (attendu 'autre_cle+0.9')")
        out[k] = (m.group(1), float(m.group(3)) * (1 if m.group(2) == "+" else -1))
    return out


def solve(order, edges, needs, anchored, relative):
    """Plus petits temps >= temps voulus (mots / relatifs) qui respectent tous les ecarts mesures."""
    deps = defaultdict(list)
    for k, d, g in list(edges) + list(needs):
        deps[k].append((d, g + (MARGIN if g > 0 else 0.0)))
    cues = {k: v for k, v in anchored.items()}
    for it in range(120):
        changed = False
        for k in order:
            pref = anchored.get(k)
            if pref is None and k in relative:
                tgt, off = relative[k]
                pref = cues[tgt] + off if tgt in cues else None
            need = max([cues[d] + g for d, g in deps[k] if d in cues], default=None)
            cand = [x for x in (pref, need) if x is not None]
            val = max(cand) if cand else 0.0
            val = max(val, 0.0)
            if abs(val - cues.get(k, -1e9)) > 1e-9:
                cues[k] = val
                changed = True
        if not changed:
            return cues, True
    return cues, False


def make_cues(ep, words_json, lead, out_path):
    order, edges, needs = measure(ep)
    ps = parse_script(ep.SCRIPT)
    asr = draft_words(ps) if words_json is None else merge_letters(load_words(words_json))
    mp = match(ps["words"], asr)
    relative = parse_relative(getattr(ep, "RELATIVE", {}))

    anchored, info = {}, {}
    unknown = [m[0] for m in ps["markers"] if m[0] not in order]
    if unknown:
        sys.exit(f"Marqueurs sans at() dans la scene : {sorted(set(unknown))}")
    bad = [k for k in relative if k not in order]
    if bad:
        sys.exit(f"RELATIVE sans at() dans la scene : {bad}")
    for key, idx, mode, off in ps["markers"]:
        t, how = word_time(idx, mp, asr, mode)
        t = max(0.0, t + off - (lead if mode == "start" else 0.0))
        anchored[key] = t
        info[key] = dict(word=ps["words"][idx], how=how + ("+end" if mode == "end" else ""), raw=t)
    for k in order:
        if k not in info:
            info[k] = dict(word="(relatif)" if k in relative else "(libre)",
                           how="relative" if k in relative else "asap", raw=None)

    cues, ok = solve(order, edges, needs, anchored, relative)
    if not ok:
        print("ATTENTION : cues relatives incompatibles avec les ecarts mesures (augmente leurs decalages).")
    rep = {}
    for k in order:
        raw = info[k]["raw"]
        if raw is None and k in relative:
            raw = cues[relative[k][0]] + relative[k][1]
        pushed = (cues[k] - raw) if raw is not None and cues[k] - raw > 0.005 else 0.0
        rep[k] = dict(word=info[k]["word"], how=info[k]["how"],
                      raw=None if raw is None else round(raw, 2), final=round(cues[k], 2),
                      pushed=round(pushed, 2))
    cues = {k: round(v, 2) for k, v in cues.items()}
    if "end" not in cues:
        sys.exit("La scene doit se terminer par self.at('end').")
    cues["end"] = round(max(cues["end"], asr[-1]["end"] + 0.3), 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"C": cues, "report": rep}, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    n_ok = sum(1 for i in range(len(ps["words"])) if i in mp)
    print(f"\nMots du script retrouves : {n_ok}/{len(ps['words'])} (transcription : {len(asr)} mots)")
    print(f"{'cle':<14}{'mot':<12}{'appariement':<13}{'t voulu':>8}{'t final':>9}  note")
    npush = 0
    for k in order:
        r = rep[k]
        note = ""
        if r["pushed"] > 0:
            note = f"repoussee de {r['pushed']:.2f}s (la voix va plus vite que l'animation)"
            npush += 1
        elif r["how"].startswith("interp"):
            note = "temps interpole : a verifier"
        raw = "" if r["raw"] is None else f"{r['raw']:.2f}"
        print(f"{k:<14}{r['word']:<12}{r['how']:<13}{raw:>8}{r['final']:>9.2f}  {note}")
    print(f"\n{npush} cle(s) repoussee(s). Si c'est trop : parle un peu moins vite a ces endroits ou allonge "
          f"la pause avant le mot.\nEcrit : {out_path}   (duree de la video : {cues['end']:.1f} s)")
    return cues


# =============================================================================================
# 6. WHISPERX
# =============================================================================================
def run_whisperx(audio, out_json, model, device, compute_type, batch_size, prompt):
    try:
        import whisperx
    except ImportError:
        sys.exit("WhisperX n'est pas installe :  pip install whisperx")
    print(f"WhisperX : modele {model} sur {device} ({compute_type})...")
    asr = whisperx.load_model(model, device, compute_type=compute_type, language="en",
                              asr_options={"initial_prompt": prompt})
    wav = whisperx.load_audio(audio)
    res = asr.transcribe(wav, batch_size=batch_size, language="en")
    am, meta = whisperx.load_align_model(language_code="en", device=device)
    res = whisperx.align(res["segments"], am, meta, wav, device, return_char_alignments=False)
    words = [{"word": w["word"], "start": w.get("start"), "end": w.get("end")}
             for seg in res["segments"] for w in seg.get("words", [])]
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(words, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(words)} mots ecrits dans {out_json}")


# =============================================================================================
# 7. BRUITAGES (synthetises a partir des evenements notes par la scene)
# =============================================================================================
SR = 44100
PENT = [0, 2, 4, 7, 9, 12, 14, 16]
BASE = 523.25  # C5


def synth_sfx(events_json, out_wav):
    from scipy.signal import fftconvolve
    data = json.load(open(events_json))
    end, events = data["end"], data["events"]
    N = int((end + 2.5) * SR)
    rng = np.random.default_rng(11)
    bus = {k: np.zeros((N, 2)) for k in ("tick", "move", "hit", "final")}

    def add(name, sig, t, pan=0.0, gain=1.0):
        i = int(t * SR)
        if i < 0 or i >= N:
            return
        sig = sig[: N - i]
        th = (pan + 1) * np.pi / 4
        bus[name][i:i + len(sig), 0] += sig * np.cos(th) * gain
        bus[name][i:i + len(sig), 1] += sig * np.sin(th) * gain

    def tvec(d):
        return np.arange(int(d * SR)) / SR

    def pluck(f, dur=0.55, tau=0.11):
        t = tvec(dur)
        s = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) + 0.1 * np.sin(6 * np.pi * f * t))
        e = np.exp(-t / tau)
        a = int(0.003 * SR)
        e[:a] *= np.linspace(0, 1, a)
        return s * e * 0.5

    def noise_lp(cut):
        n = len(cut)
        x = rng.standard_normal(n)
        a = 1 - np.exp(-2 * np.pi * cut / SR)
        y1 = y2 = 0.0
        out = np.empty(n)
        for i in range(n):
            y1 += a[i] * (x[i] - y1)
            y2 += a[i] * (y1 - y2)
            out[i] = y2
        m = np.max(np.abs(out))
        return out / m if m > 0 else out

    def whoosh(d, f0, f1):
        u = np.linspace(0, 1, int(d * SR))
        return noise_lp(f0 * (f1 / f0) ** np.sin(np.pi * u)) * np.sin(np.pi * u) ** 2

    def thud():
        t = tvec(0.55)
        f = 42 + 53 * np.exp(-t / 0.06)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18) + \
            rng.standard_normal(len(t)) * np.exp(-t / 0.008) * 0.25

    def clack():
        t = tvec(0.09)
        n = np.diff(rng.standard_normal(len(t)) * np.exp(-t / 0.010), prepend=0.0)
        return 0.6 * n + 0.8 * np.sin(2 * np.pi * 150 * t) * np.exp(-t / 0.025)

    def crackle(d, density):
        s = np.zeros(int(d * SR))
        for _ in range(int(density * d)):
            i = rng.integers(0, len(s) - 200)
            k = np.arange(200) / SR
            s[i:i + 200] += rng.choice([-1, 1]) * rng.uniform(0.3, 1.0) * np.exp(-k / 0.0015)
        return np.diff(s, prepend=0.0)

    def rise(d):
        u = np.linspace(0, 1, int(d * SR))
        nz = noise_lp(300 * (7000 / 300) ** u) * u ** 2.2
        sw = np.sin(2 * np.pi * np.cumsum(220 * (1100 / 220) ** u) / SR) * u ** 2.2 * 0.25
        return nz + sw

    def impact():
        t = tvec(2.2)
        sub = np.sin(2 * np.pi * np.cumsum(38 + 60 * np.exp(-t / 0.08)) / SR) * np.exp(-t / 0.55)
        return sub + noise_lp(np.full(len(t), 900.0)) * np.exp(-t / 0.12) * 0.8

    idx, n_tick = 2, 0
    for e in events:
        k, t, g = e["kind"], e["t"], e.get("gain")
        if k == "tick":
            step = int(rng.choice([-2, -1, 1, 1, 2]))
            idx += step
            if idx < 0 or idx >= len(PENT):
                idx = max(0, min(len(PENT) - 1, idx - 2 * step))
            n_tick += 1
            add("tick", pluck(BASE * 2 ** (PENT[idx] / 12)), t, -0.3 if n_tick % 2 else 0.3, g or 1.0)
        elif k == "ping":
            add("tick", pluck(BASE * 4, 0.9, 0.28), t, 0.0, g or 0.8)
        elif k == "lock":
            add("tick", pluck(BASE * 1.5), t, -0.2, 0.9)
            add("tick", pluck(BASE * 2.0), t + 0.09, 0.2, 0.9)
        elif k == "clack":
            add("hit", clack(), t, 0.0, g or 0.55)
        elif k == "thud":
            add("hit", thud(), t, 0.0, (g or 0.9) * 0.9)
        elif k == "whoosh":
            add("move", whoosh(e.get("dur", 1.0), e.get("f0", 250), e.get("f1", 1400)), t, e.get("pan", 0.0),
                g or 0.5)
        elif k == "crackle":
            add("move", crackle(e.get("dur", 0.5), e.get("density", 80)), t, 0.0, g or 0.4)
        elif k == "rise":
            add("final", rise(e["dur"]), t, 0.0, g or 0.55)
        elif k == "impact":
            add("final", impact(), t, 0.0, g or 1.0)

    ir_s = rng.standard_normal(int(0.45 * SR)) * np.exp(-np.arange(int(0.45 * SR)) / (0.12 * SR))
    ir_l = rng.standard_normal(int(1.8 * SR)) * np.exp(-np.arange(int(1.8 * SR)) / (0.5 * SR))

    def wet(x, ir, amount):
        o = np.zeros_like(x)
        for c in (0, 1):
            o[:, c] = fftconvolve(x[:, c], ir / np.sqrt(np.sum(ir ** 2)))[:N] * amount
        return o

    mix = (bus["tick"] * 0.26 + wet(bus["tick"], ir_s, 0.10) + bus["move"] * 0.12 + bus["hit"] * 0.30
           + bus["final"] * 0.60 + wet(bus["final"], ir_l, 0.12))
    fade = int(0.05 * SR)
    mix[:fade] *= np.linspace(0, 1, fade)[:, None]
    e0 = int(end * SR)
    mix[e0:] *= np.exp(-np.arange(N - e0) / (0.35 * SR))[:, None]
    mix = mix / max(np.max(np.abs(mix)), 1e-9) * 0.8
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"Bruitages : {out_wav}  ({len(events)} evenements, {N / SR:.1f} s)")


# =============================================================================================
# 8. CLI
# =============================================================================================
def paths(ep_file):
    here = os.path.dirname(os.path.abspath(ep_file))
    stem = os.path.splitext(os.path.basename(ep_file))[0]
    b = os.path.join(here, "build")
    return dict(here=here, stem=stem, build=b, cues=os.path.join(b, "cues.json"),
                words=os.path.join(b, "words.json"), events=os.path.join(b, "sfx_events.json"),
                sfx=os.path.join(b, "sfx.wav"), media=os.path.join(b, "media"))


def run(cmd, **kw):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def whisper_prompt(ep):
    return getattr(ep, "WHISPER_PROMPT", "")


def cmd_gaps(a):
    ep = load_episode(a.episode)
    order, edges, needs = measure(ep)
    print(f"{len(order)} cues, dans l'ordre de la timeline.\n")
    print("Ecarts minimaux mesures (cle >= precedente + ecart) :")
    by = defaultdict(list)
    for k, d, g in edges:
        by[k].append((d, g))
    for k in order:
        for d, g in by[k]:
            if g > 0:
                print(f"  {k:<14} >= {d:<14} + {g + MARGIN:5.2f} s")
    for k, d, g in needs:
        print(f"  {k:<14} >= {d:<14} + {g + MARGIN:5.2f} s   (need)")


def cmd_script(a):
    ep = load_episode(a.episode)
    p = paths(a.episode)
    txt = reading_text(ep.SCRIPT)
    os.makedirs(p["build"], exist_ok=True)
    out = os.path.join(p["build"], f"{p['stem']}_lecture.txt")
    open(out, "w", encoding="utf-8").write(txt + "\n")
    print(txt + f"\n\n(ecrit dans {out})")


def cmd_draft(a):
    ep = load_episode(a.episode)
    make_cues(ep, None, a.lead, paths(a.episode)["cues"])


def cmd_transcribe(a):
    """WhisperX seul (pas besoin que la scene existe) : mots + temps, et controle de lecture du script."""
    ep = load_episode(a.episode)
    p = paths(a.episode)
    if not a.voice:
        sys.exit("Donne la voix :  python3 guidage.py %s transcribe --voice voix.wav" % os.path.basename(a.episode))
    run_whisperx(os.path.abspath(a.voice), p["words"], a.model, a.device, a.compute_type, a.batch_size,
                 whisper_prompt(ep))
    ws = load_words(p["words"])
    txt = os.path.join(p["build"], f"{p['stem']}_mots.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("debut\tfin\tmot\n")
        for w in ws:
            f.write(f"{w['start']:.2f}\t{w['end']:.2f}\t{w['word']}\n")
    ps = parse_script(ep.SCRIPT)
    mp = match(ps["words"], merge_letters(ws))
    miss = [ps["words"][i] for i in range(len(ps["words"])) if i not in mp]
    print(f"\nMots du script retrouves dans ta voix : {len(ps['words']) - len(miss)}/{len(ps['words'])}")
    if miss:
        print("Non retrouves (oubli, mot change ou mal transcrit) :", ", ".join(miss))
    print(f"Duree parlee : {ws[-1]['end']:.1f} s")

    # liste courte : une ligne par marqueur [[cle]] du script (a coller dans la conversation)
    asr = merge_letters(ws)
    lines = []
    for key, idx, mode, off in ps["markers"]:
        t, how = word_time(idx, mp, asr, mode)
        lines.append(f"{key:<10} {t + off:7.2f}  {ps['words'][idx]}" + ("  (estime)" if how == "interp" else ""))
    cles = os.path.join(p["build"], f"{p['stem']}_cles.txt")
    open(cles, "w", encoding="utf-8").write("\n".join(lines) + f"\nfin_voix  {ws[-1]['end']:7.2f}\n")
    print("\n--- A COPIER-COLLER DANS LA CONVERSATION (cle  temps en s  mot) ---")
    print("\n".join(lines))
    print(f"fin_voix  {ws[-1]['end']:7.2f}")
    print("--- fin ---")
    print(f"\nLes memes lignes sont dans {cles}. L'audio n'est jamais necessaire : seuls ces temps comptent.")


def cmd_sync(a):
    ep = load_episode(a.episode)
    p = paths(a.episode)
    if a.voice and not a.skip_whisper:
        run_whisperx(a.voice, p["words"], a.model, a.device, a.compute_type, a.batch_size, whisper_prompt(ep))
    if not os.path.exists(p["words"]):
        sys.exit("Pas de build/words.json : donne --voice voix.wav")
    make_cues(ep, p["words"], a.lead, p["cues"])


def cmd_build(a):
    ep = load_episode(a.episode)
    p = paths(a.episode)
    os.makedirs(p["build"], exist_ok=True)
    voice = os.path.abspath(a.voice) if a.voice else None
    if voice:
        if not a.skip_whisper:
            run_whisperx(voice, p["words"], a.model, a.device, a.compute_type, a.batch_size, whisper_prompt(ep))
        elif not os.path.exists(p["words"]):
            sys.exit("--skip-whisper demande mais build/words.json n'existe pas")
        make_cues(ep, p["words"], a.lead, p["cues"])
    else:
        print("Pas de voix : cues estimees (brouillon).")
        make_cues(ep, None, a.lead, p["cues"])
    end_t = json.load(open(p["cues"]))["C"]["end"]

    env = dict(os.environ, PREVIEW="1" if a.preview else "0", GUIDAGE_CUES=p["cues"],
               GUIDAGE_SFX_EVENTS=p["events"])
    cls = getattr(ep, "SCENE", "Episode")
    if not a.skip_render:
        shutil.rmtree(p["media"], ignore_errors=True)
        run([sys.executable, "-m", "manim", "-ql" if a.preview else "-qh", "--disable_caching",
             "--media_dir", p["media"], os.path.join(p["here"], p["stem"] + ".py"), cls], env=env)
    vids = sorted(glob.glob(os.path.join(p["media"], "videos", p["stem"], "*", cls + ".mp4")),
                  key=os.path.getmtime)
    if not vids:
        sys.exit("Video manim introuvable (relance sans --skip-render).")
    synth_sfx(p["events"], p["sfx"])

    out = os.path.join(p["build"], f"{p['stem']}_{'apercu' if a.preview else 'final'}.mp4")
    if voice:
        flt = ("[1:a]aformat=sample_rates=44100:channel_layouts=stereo,highpass=f=80,"
               "acompressor=threshold=-20dB:ratio=3:attack=5:release=100:makeup=3,apad[v];"
               "[2:a]aformat=sample_rates=44100:channel_layouts=stereo[s];[v]asplit=2[v1][v2];"
               "[s][v1]sidechaincompress=threshold=0.02:ratio=6:attack=15:release=300[sd];"
               "[v2][sd]amix=inputs=2:duration=longest:normalize=0[m];"
               f"[m]loudnorm=I=-14:TP=-1.5:LRA=11,atrim=0:{end_t},asetpts=PTS-STARTPTS[a]")
        inputs = ["-i", vids[-1], "-i", voice, "-i", p["sfx"]]
    else:
        flt = f"[1:a]atrim=0:{end_t},asetpts=PTS-STARTPTS[a]"
        inputs = ["-i", vids[-1], "-i", p["sfx"]]
    run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", flt, "-map", "0:v",
         "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", str(end_t), out])
    print(f"\nTermine : {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="fichier d'episode, ex. ep01.py")
    ap.add_argument("command", choices=["script", "gaps", "draft", "transcribe", "sync", "build"])
    ap.add_argument("--voice", help="enregistrement de la voix off (wav/mp3/m4a)")
    ap.add_argument("--preview", action="store_true", help="rendu 540p/15 fps")
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lead", type=float, default=0.05, help="avance (s) des apparitions sur le debut du mot")
    a = ap.parse_args()
    {"script": cmd_script, "gaps": cmd_gaps, "draft": cmd_draft, "transcribe": cmd_transcribe,
     "sync": cmd_sync, "build": cmd_build}[a.command](a)


if __name__ == "__main__":
    main()
