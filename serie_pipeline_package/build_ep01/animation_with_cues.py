
from manim import *
import numpy as np
import os

PREVIEW = os.environ.get("PREVIEW") == "1"
PW = 540 if PREVIEW else 1080
config.pixel_width = PW
config.pixel_height = PW * 16 // 9
config.frame_width = 9.0
config.frame_height = 16.0
config.frame_rate = 15 if PREVIEW else 30  # 30 fps pour le rendu final (synchro audio)

# ------------------------------------------------------------------ charte Blueprint
BG = "#0b2a5b"      # fond
INK = "#f4f8ff"     # structure
CYAN = "#8fe3ff"    # information active
AMBER = "#ffcf5a"   # conclusion (une seule fois)
GRIDC = "#5b83bd"
FONT = "DejaVu Sans Mono"
config.background_color = BG

# ------------------------------------------------------------------ cues voix (s)
C = dict(
    hook2=1.0, hook_out=2.0,
    no_comp=2.4, no_sens=3.3, no_steer=4.2, physics=5.3,
    release=6.0, g_arrow=6.2, v_arrow=8.0, curve=9.2, ballistic=9.6, land=10.6,
    air=11.4, drag=11.8, wind=13.3, gap=14.9, error=15.3, grows=16.4,
    to_air=17.3, aircraft=19.0, computer=19.8, in_speed=21.0, in_alt=21.9, in_wind=22.8,
    solves=23.8,
    back=25.0, ccip=25.8, impact_pt=26.6, shows=28.2, now=29.0,
    move=30.4, target=32.4, put=33.4, rel_ccip=34.0,
    ccrp=36.3, ccrp_flight=37.3, fly=37.8, ccrp_target=39.6, ccrp_mark=40.8,
    ccrp_label=41.8, ccrp_rel=42.2,
    rocket=44.1, no_steer2=46.2, pred_rocket=47.4, fire=48.4,
    prediction=50.6, release_word=51.1, go=51.5, no_corr=52.5,
    eq_clear=54.1, eq1=54.4, eq2=55.0, eq3=55.5, eq4=56.1, eq5=56.7, eq_box=57.0,
    nxt=57.6, loop=59.6,
)
END_T = 60.6

# Horaires generes automatiquement par serie_pipeline.py
_cues_json_path = os.environ.get("MANIM_CUES_JSON")
if _cues_json_path:
    import json as _json
    with open(_cues_json_path, "r", encoding="utf-8") as _cue_file:
        _generated_cues = _json.load(_cue_file)
    C.update({k: float(v) for k, v in _generated_cues["C"].items()})
    END_T = float(_generated_cues["END_T"])


# ------------------------------------------------------------------ geometrie / physique (illustratif)
AY, GY = 2.2, -3.7           # altitude de l'avion / sol
XA0 = -3.0                   # point de largage "naif"
START_DY = -0.30             # la bombe est sous l'avion
SX = 2.8                     # vitesse horizontale (unites/s)
G = 3.05                     # gravite
DRAG_K = 0.05                # trainee quadratique
WIND = -0.9                  # vent de face
GUST = (0.7, (-2.4, 0.0))    # rafale apres largage : (instant, vecteur vent ajoute)
DT = 0.008
EQ_X0 = -3.8                 # alignement gauche des textes
ROW_Y0, ROW_DY = 4.9, 0.44


def simulate(p0, v0, k=0.0, wind=(0.0, 0.0), gust=None, dt=DT):
    p = np.array(p0, float)[:2]
    v = np.array(v0, float)[:2]
    ground = GY + 0.05
    pts, t = [p.copy()], 0.0
    while len(pts) < 5000:
        w = np.array(wind, float)
        if gust and t >= gust[0]:
            w = w + np.array(gust[1], float)
        rel = v - w
        a = np.array([0.0, -G]) - k * np.linalg.norm(rel) * rel
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


P0 = (XA0, AY + START_DY)
P_VAC = simulate(P0, (SX, 0))                                   # gravite seule
P_DRAG = simulate(P0, (SX, 0), k=DRAG_K)                        # + trainee
P_WIND = simulate(P0, (SX, 0), k=DRAG_K, wind=(WIND, 0))        # + vent
XT = P_VAC[-1][0]                                               # cible = impact "ideal"
R_REAL = P_WIND[-1][0] - XA0                                    # portee reelle
XA1 = XT - R_REAL                                               # bon point de largage
P_GUST = simulate((XA1, AY + START_DY), (SX, 0), k=DRAG_K, wind=(WIND, 0), gust=GUST)
GUST_IDX = int(GUST[0] / DT)


def solve_rocket(speed=9.0, k=0.01):
    """Angle de tir (rad) pour que la roquette touche XT."""
    lo, hi = -1.45, -0.05
    for _ in range(45):
        mid = (lo + hi) / 2
        end = simulate(P0, (speed * np.cos(mid), speed * np.sin(mid)), k=k)[-1][0]
        if end < XT:
            lo = mid
        else:
            hi = mid
    a = (lo + hi) / 2
    return simulate(P0, (speed * np.cos(a), speed * np.sin(a)), k=k)


P_ROCKET = solve_rocket()


def path_from(P, x):
    """Meme trajectoire, largage depuis l'abscisse x."""
    return P - P[0] + np.array([x, AY + START_DY, 0.0])


# ------------------------------------------------------------------ outils graphiques
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


def fit_right(m, xmax=3.45):
    if m.get_right()[0] > xmax:
        m.scale((xmax - m.get_left()[0]) / m.width, about_point=m.get_left())
    return m


def aircraft_shape():
    body = [(0.70, 0.00), (0.42, 0.07), (0.20, 0.16), (0.00, 0.16), (-0.12, 0.08), (-0.40, 0.08),
            (-0.58, 0.40), (-0.72, 0.40), (-0.68, 0.06), (-0.74, -0.02), (-0.56, -0.11), (0.38, -0.11)]
    poly = Polygon(*[[x, y, 0] for x, y in body], stroke_color=INK, stroke_width=2.6,
                   fill_color=BG, fill_opacity=1)
    wing = Polygon([0.18, -0.02, 0], [-0.22, -0.02, 0], [-0.40, -0.17, 0], [-0.14, -0.17, 0],
                   fill_opacity=0)
    canopy = Line([0.30, 0.09, 0], [0.02, 0.09, 0])
    det = VGroup(wing, canopy).set_stroke(INK, 1.4, opacity=0.75)
    return VGroup(poly, det).scale(1.15)


def bomb_shape():
    R = [(0.32, 0.0), (0.26, 0.055), (0.10, 0.085), (-0.16, 0.085), (-0.26, 0.15),
         (-0.34, 0.15), (-0.30, 0.04), (-0.34, 0.0)]
    pts = R + [(x, -y) for x, y in reversed(R[1:-1])]
    poly = Polygon(*[[x, y, 0] for x, y in pts], stroke_color=INK, stroke_width=2.4,
                   fill_color=BG, fill_opacity=1)
    det = VGroup(Line([0.06, -0.085, 0], [0.06, 0.085, 0]),
                 Line([-0.16, -0.085, 0], [-0.16, 0.085, 0])).set_stroke(INK, 1.3, opacity=0.75)
    return VGroup(poly, det)


def rocket_shape():
    R = [(0.42, 0.0), (0.36, 0.045), (0.22, 0.06), (-0.28, 0.06), (-0.36, 0.13),
         (-0.44, 0.13), (-0.40, 0.03), (-0.44, 0.0)]
    pts = R + [(x, -y) for x, y in reversed(R[1:-1])]
    poly = Polygon(*[[x, y, 0] for x, y in pts], stroke_color=INK, stroke_width=2.4,
                   fill_color=BG, fill_opacity=1)
    det = VGroup(Line([0.14, -0.06, 0], [0.14, 0.06, 0]),
                 Line([-0.1, -0.06, 0], [-0.1, 0.06, 0])).set_stroke(INK, 1.3, opacity=0.75)
    return VGroup(poly, det)


def target_at(x):
    return VGroup(Circle(radius=0.26, color=INK, stroke_width=2.5),
                  *[Line(d * 0.18, d * 0.48, color=INK, stroke_width=2.2)
                    for d in (UP, DOWN, LEFT, RIGHT)]).move_to([x, GY + 0.5, 0]).set_z_index(4)


def ground_cote(xa, xb, label="Δ"):
    lo, hi = sorted((xa, xb))
    yc = GY + 0.85
    ex = VGroup(*[DashedLine([x, GY + 0.12, 0], [x, yc + 0.2, 0], dash_length=0.08,
                             stroke_width=1.2, color=CYAN) for x in (xa, xb)])
    ar = DoubleArrow([lo, yc, 0], [hi, yc, 0], buff=0, stroke_width=2.4, tip_length=0.16,
                     color=CYAN, max_tip_length_to_length_ratio=0.45)
    lb = TX(label, 36, CYAN).move_to([(lo + hi) / 2, yc + 0.38, 0])
    return VGroup(ex, ar, lb).set_z_index(5)


def vec(p, d, length):
    d = np.array(d, float)
    d = d / np.linalg.norm(d)
    return Arrow(p, p + d * length, buff=0, color=CYAN, stroke_width=2.6, tip_length=0.16,
                 max_tip_length_to_length_ratio=0.5)


class Flight:
    """Trajectoire pilotee par le temps de la scene : t0 = debut, dur = duree visuelle."""

    def __init__(self, scene, pts, t0, dur, color=CYAN, width=2.4):
        self.s, self.pts, self.t0, self.dur = scene, pts, t0, dur
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


class Unguided(Scene):
    # ---------------------------------------------------------------- synchro
    def at(self, t):
        d = t - self.time
        frame = 1.0 / config.frame_rate
        if d > frame:
            self.wait(d)
        elif d < -frame:
            self.late.append((round(t, 2), round(-d, 3)))

    def show(self, *mobs, rt=0.35):
        self.play(*[FadeIn(m) for m in mobs], run_time=rt)

    # ---------------------------------------------------------------- mouvement de l'avion (piloté par le temps)
    def ax_val(self, t):
        x = XA0
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

    def glide(self, xb, t0, t1, smooth=True):
        self.segs.append((t0, t1, self.ax_val(t0), xb, smooth))

    def start_flight(self, pts, t0, dur):
        f = Flight(self, pts, t0, dur)
        self.fl = f
        self.add(f.trail)
        return f

    def make_proj(self, shape):
        shape.set_z_index(6)
        shape.ang = 0.0

        def up(m, dt):
            if self.fl is None:
                p = np.array([self.ax_val(self.time), AY + START_DY, 0.0])
                ang = 0.0
            else:
                p, ang = self.fl.pos(), self.fl.angle()
            m.rotate(ang - m.ang, about_point=m.get_center())
            m.ang = ang
            m.move_to(p)
        shape.add_updater(up)
        up(shape, 0)
        return shape

    def make_follow(self, kind, tracker):
        g = VGroup().set_z_index(7)

        def up(m, dt):
            a = tracker.get_value()
            if self.fl is None or a < 0.02:
                m.become(VGroup())
                return
            p, ang = self.fl.pos(), self.fl.angle()
            d = np.array([np.cos(ang), np.sin(ang), 0.0])
            if kind == "g":
                ar = vec(p + DOWN * 0.15, DOWN, 0.85)
                lb = TX("g", 28, CYAN).move_to(p + np.array([0.3, -0.6, 0]))
            elif kind == "v":
                ar = vec(p + d * 0.35, d, 1.0)
                lb = TX("v", 28, CYAN).move_to(p + d * 1.5 + UP * 0.28)
            else:
                ar = vec(p - d * 0.35, -d, 0.8)
                lb = TX("D", 28, CYAN).move_to(p - d * 1.45 + UP * 0.28)
            m.become(VGroup(ar, lb).set_opacity(a))
            m.set_z_index(7)
        g.add_updater(up)
        return g

    def make_pred(self, x):
        pts = path_from(P_WIND, x)
        pts = np.vstack([pts[::4], pts[-1:]])
        curve = VMobject().set_points_as_corners(pts)
        d = DashedVMobject(curve, num_dashes=46, dashed_ratio=0.55).set_stroke(CYAN, 2.0).set_z_index(2)
        d.last_x = x

        def up(m, dt):
            nx = self.ax_val(self.time)
            m.shift(RIGHT * (nx - m.last_x))
            m.last_x = nx
        d.add_updater(up)
        return d

    def pipper_follow(self):
        p = VGroup(Circle(radius=0.15, color=CYAN, stroke_width=2.2),
                   Dot(radius=0.035, color=CYAN)).set_z_index(6)
        p.move_to([XA0 + R_REAL, GY + 0.5, 0])
        p.add_updater(lambda m, dt: m.move_to([self.ax_val(self.time) + R_REAL, GY + 0.5, 0]))
        return p

    def rowt(self, s, i, size=24, color=INK):
        return TX(s, size, color).move_to([EQ_X0, ROW_Y0 - ROW_DY * i, 0], aligned_edge=LEFT).set_z_index(9)

    def ping(self, pos):
        return Flash(pos, color=CYAN, flash_radius=0.4, line_length=0.14, num_lines=10,
                     line_stroke_width=2.2, run_time=0.5)

    # ---------------------------------------------------------------- scene
    def construct(self):
        self.late = []
        self.segs = []
        self.fl = None
        ga, va, da = ValueTracker(0), ValueTracker(0), ValueTracker(0)

        # fond blueprint
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
        sweep = Line([-4.25, 0, 0], [4.25, 0, 0], color=CYAN, stroke_width=1.6,
                     stroke_opacity=0.10).set_z_index(1)
        sweep.add_updater(lambda m, dt: m.move_to([0, 8 - 16 * ((self.time / (END_T / 8)) % 1), 0]))

        ground = VGroup(
            Line([-4.25, GY, 0], [4.25, GY, 0], color=INK, stroke_width=2.2),
            *[Line([x, GY, 0], [x - 0.2, GY - 0.2, 0], color=INK, stroke_width=1.0, stroke_opacity=0.45)
              for x in np.arange(-4.0, 4.3, 0.35)]).set_z_index(1)

        # avion, bombe, roquette
        air = aircraft_shape().set_z_index(5).move_to([XA0, AY, 0])
        air.add_updater(lambda m, dt: m.move_to([self.ax_val(self.time), AY, 0]))
        bomb = self.make_proj(bomb_shape())
        rocket = self.make_proj(rocket_shape())
        g_g = self.make_follow("g", ga)
        v_g = self.make_follow("v", va)
        d_g = self.make_follow("d", da)

        # accroche en deux temps (derniere image = premiere image)
        hook = TX("THE BOMB KNOWS", 44, INK).move_to([0, 5.2, 0])
        hook_rule = Line([-3.0, 4.72, 0], [3.0, 4.72, 0], color=CYAN, stroke_width=2.0)
        hook_group = VGroup(hook, hook_rule).set_z_index(10)
        hook2 = TX("NOTHING", 44, CYAN).move_to([0, 4.2, 0]).set_z_index(10)

        self.add(grid, border, sweep, ground, air, bomb, hook_group, g_g, v_g, d_g)

        # ============================================================ 0 - ACCROCHE
        self.at(C["hook2"])
        self.play(FadeIn(hook2), run_time=0.4)
        self.at(C["hook_out"])
        self.play(FadeOut(hook_group), FadeOut(hook2), run_time=0.3)

        # ============================================================ 1 - "NO COMPUTER / SENSOR / STEERING"
        r1, r2, r3 = (self.rowt("NO COMPUTER", 0, 26), self.rowt("NO SENSOR", 1, 26),
                      self.rowt("NO STEERING", 2, 26))
        self.at(C["no_comp"])
        self.play(FadeIn(r1), self.ping(bomb.get_center()), run_time=0.4)
        self.at(C["no_sens"])
        self.play(FadeIn(r2), self.ping(bomb.get_center()), run_time=0.4)
        self.at(C["no_steer"])
        self.play(FadeIn(r3), self.ping(bomb.get_center()), run_time=0.4)
        ph = TX("PHYSICS", 30, CYAN).move_to([EQ_X0, ROW_Y0, 0], aligned_edge=LEFT).set_z_index(9)
        self.at(C["physics"])
        self.play(FadeOut(VGroup(r1, r2, r3)), FadeIn(ph), run_time=0.4)

        # ============================================================ 2 - TRAJECTOIRE BALISTIQUE (gravite + vitesse)
        self.at(C["release"])
        V = self.start_flight(P_VAC, C["release"], C["land"] - C["release"])
        self.at(C["g_arrow"])
        self.play(ga.animate.set_value(1), run_time=0.4)
        self.at(C["v_arrow"])
        self.play(va.animate.set_value(1), run_time=0.4)
        self.at(C["curve"])
        self.play(ga.animate.set_value(0), va.animate.set_value(0), run_time=0.4)
        r_bt = TX("BALLISTIC TRAJECTORY", 24, CYAN).move_to(
            [EQ_X0, ROW_Y0 - ROW_DY, 0], aligned_edge=LEFT).set_z_index(9)
        self.at(C["ballistic"])
        self.show(r_bt)
        tgt = target_at(XT)
        self.at(C["land"])
        self.play(FadeIn(tgt, scale=1.4), run_time=0.4)
        self.play(FadeOut(VGroup(ph, r_bt)), FadeOut(bomb),
                  V.trail.animate.set_stroke(opacity=0.3), run_time=0.35)
        self.fl = None
        bomb.update(0)

        # ============================================================ 3 - L'AIR DEVIE LA COURBE
        self.at(C["air"])
        self.play(FadeIn(bomb), run_time=0.3)
        r_drag, r_wind, r_err = (self.rowt("DRAG", 0, 28), self.rowt("WIND", 1, 28),
                                 self.rowt("ERROR", 2, 28))
        self.at(C["drag"])
        D = self.start_flight(P_DRAG, C["drag"], 1.0)
        self.play(FadeIn(r_drag), da.animate.set_value(1), run_time=0.3)
        self.at(C["drag"] + 1.0)
        self.play(da.animate.set_value(0), FadeOut(bomb),
                  D.trail.animate.set_stroke(opacity=0.3), run_time=0.2)
        self.fl = None
        bomb.update(0)
        self.play(FadeIn(bomb), run_time=0.2)

        wind_g = VGroup(*[Arrow([3.1, y, 0], [1.7, y, 0], buff=0, color=CYAN, stroke_width=2.2,
                                tip_length=0.14) for y in (1.9, 1.5, 1.1)]).set_z_index(5)
        self.at(C["wind"])
        W = self.start_flight(P_WIND, C["wind"], 1.2)
        self.play(FadeIn(r_wind), FadeIn(wind_g), run_time=0.4)

        gap1 = ground_cote(P_WIND[-1][0], XT)
        self.at(C["gap"])
        self.play(FadeIn(gap1), run_time=0.4)
        self.at(C["error"])
        self.play(FadeIn(r_err), run_time=0.35)
        self.at(C["grows"])
        self.play(Indicate(gap1[2], color=CYAN, scale_factor=1.4), run_time=0.5)

        # ============================================================ 4 - L'INTELLIGENCE EST DANS L'AVION
        self.at(C["to_air"])
        self.play(FadeOut(VGroup(r_drag, r_wind, r_err)), FadeOut(wind_g), FadeOut(bomb), FadeOut(gap1),
                  FadeOut(V.trail), FadeOut(D.trail), FadeOut(W.trail), run_time=0.4)
        self.fl = None
        self.glide(XA1, C["to_air"] + 0.5, C["to_air"] + 1.2)
        self.at(C["to_air"] + 1.2)
        bomb.update(0)
        self.play(FadeIn(bomb), run_time=0.3)

        names = ["SPEED", "ALTITUDE", "WIND"]
        in_rows = [self.rowt(s_, i_, 22) for i_, s_ in enumerate(names)]
        cyc = ROW_Y0 - ROW_DY
        c_blk = block("COMPUTER", 2.2, 1.4, 22).move_to([0, cyc, 0]).set_z_index(9)
        in_arrows = [arr([-1.95, ROW_Y0 - i_ * ROW_DY, 0], [-1.15, ROW_Y0 - i_ * ROW_DY, 0]).set_z_index(9)
                     for i_ in range(3)]
        out_ar = arr([1.1, cyc, 0], [1.6, cyc, 0]).set_z_index(9)
        out_lb = fit_right(TX("TRAJECTORY", 22, CYAN).move_to([1.7, cyc, 0], aligned_edge=LEFT)).set_z_index(9)
        pred = self.make_pred(XA1)

        self.at(C["aircraft"])
        self.play(self.ping(air.get_center()), run_time=0.5)
        self.at(C["computer"])
        self.show(c_blk, rt=0.4)
        for cue, i_ in ((C["in_speed"], 0), (C["in_alt"], 1), (C["in_wind"], 2)):
            self.at(cue)
            self.play(FadeIn(in_rows[i_]), FadeIn(in_arrows[i_]), run_time=0.35)
        self.at(C["solves"])
        self.play(FadeIn(out_ar), FadeIn(out_lb), Create(pred), run_time=0.8)

        # ============================================================ 7 - CCIP
        self.at(C["back"])
        self.glide(XA0, C["back"], C["back"] + 0.9)
        self.play(FadeOut(VGroup(*in_rows, *in_arrows, c_blk, out_ar, out_lb)), run_time=0.5)
        rc = [self.rowt("CCIP", 0, 28), self.rowt("IF RELEASED NOW", 1, 24), self.rowt("TARGET", 2, 24)]
        pipper = self.pipper_follow()

        self.at(C["ccip"])
        self.show(rc[0])
        self.at(C["impact_pt"])
        self.play(FadeIn(pipper), run_time=0.4)
        self.at(C["shows"])
        self.play(self.ping(pipper.get_center()), run_time=0.5)
        self.at(C["now"])
        self.show(rc[1])
        self.at(C["move"])
        self.glide(XA1, C["move"], C["put"])
        self.at(C["target"])
        self.show(rc[2])
        self.at(C["put"])
        self.play(Indicate(tgt, color=CYAN, scale_factor=1.15), run_time=0.5)
        self.at(C["rel_ccip"])
        F1 = self.start_flight(path_from(P_WIND, XA1), C["rel_ccip"], 1.3)
        self.play(FadeOut(pred), FadeOut(pipper), run_time=0.3)
        self.at(C["rel_ccip"] + 1.3)
        self.glide(XA0, C["rel_ccip"] + 1.4, C["rel_ccip"] + 2.1)
        self.play(self.ping(tgt.get_center()), FadeOut(VGroup(*rc)), FadeOut(F1.trail),
                  FadeOut(bomb), run_time=0.5)
        self.fl = None

        # ============================================================ 8 - CCRP
        rr = [self.rowt("CCRP", 0, 28), self.rowt("TARGET", 1, 24), self.rowt("RELEASE POINT", 2, 24)]
        fline = DashedLine([-4.0, AY, 0], [3.4, AY, 0], dash_length=0.12, color=INK,
                           stroke_width=1.5, stroke_opacity=0.4).set_z_index(2)
        rel_mark = VGroup(Line([XA1, AY - 0.65, 0], [XA1, AY + 0.65, 0], color=CYAN, stroke_width=2.0),
                          Triangle(color=CYAN, fill_color=CYAN, fill_opacity=1, stroke_width=1)
                          .scale(0.12).rotate(PI).move_to([XA1, AY + 0.82, 0])).set_z_index(3)
        self.at(C["ccrp"])
        self.show(rr[0])
        self.at(C["ccrp"] + 0.6)
        bomb.update(0)
        self.play(FadeIn(bomb), run_time=0.3)
        self.at(C["ccrp_flight"])
        self.show(fline, rt=0.4)
        self.at(C["fly"])
        self.glide(XA1, C["fly"], C["ccrp_rel"], smooth=False)
        self.at(C["ccrp_target"])
        self.play(FadeIn(rr[1]), Indicate(tgt, color=CYAN, scale_factor=1.15), run_time=0.4)
        self.at(C["ccrp_mark"])
        self.show(rel_mark, rt=0.4)
        self.at(C["ccrp_label"])
        self.show(rr[2])
        self.at(C["ccrp_rel"])
        F2 = self.start_flight(path_from(P_WIND, XA1), C["ccrp_rel"], 1.3)
        self.at(C["ccrp_rel"] + 1.3)
        t_end = C["ccrp_rel"] + 1.3
        self.glide(XA0, t_end + 0.1, t_end + 0.9)
        self.play(self.ping(tgt.get_center()), FadeOut(VGroup(*rr, rel_mark, fline)),
                  FadeOut(F2.trail), FadeOut(bomb), run_time=0.5)
        self.fl = None

        # ============================================================ 9 - ROQUETTES
        rk = [self.rowt("ROCKET", 0, 28), self.rowt("NO STEERING", 1, 24), self.rowt("IMPACT POINT", 2, 24)]
        pred_r = DashedVMobject(
            VMobject().set_points_as_corners(np.vstack([P_ROCKET[::2], P_ROCKET[-1:]])),
            num_dashes=36, dashed_ratio=0.55).set_stroke(CYAN, 2.0).set_z_index(2)
        pipper2 = VGroup(Circle(radius=0.15, color=CYAN, stroke_width=2.2),
                         Dot(radius=0.035, color=CYAN)).set_z_index(6).move_to([XT, GY + 0.5, 0])
        self.at(C["rocket"])
        self.show(rk[0])
        self.at(C["rocket"] + 0.7)
        rocket.update(0)
        self.play(FadeIn(rocket), run_time=0.3)
        self.at(C["no_steer2"])
        self.show(rk[1])
        self.at(C["pred_rocket"])
        self.play(FadeIn(rk[2]), Create(pred_r), FadeIn(pipper2), run_time=0.8)
        self.at(C["fire"])
        F3 = self.start_flight(P_ROCKET, C["fire"], 1.1)
        self.at(C["fire"] + 1.1)
        t_end = C["fire"] + 1.1
        self.glide(XA1, t_end + 0.1, t_end + 0.7)
        self.play(self.ping(tgt.get_center()), FadeOut(VGroup(*rk)), FadeOut(pred_r), FadeOut(pipper2),
                  FadeOut(F3.trail), FadeOut(rocket), run_time=0.5)
        self.fl = None

        # ============================================================ 10 - LA PREDICTION S'ARRETE AU LARGAGE
        self.at(t_end + 0.7)
        bomb.update(0)
        self.play(FadeIn(bomb), run_time=0.3)
        self.at(C["prediction"])
        self.play(FadeIn(pred), run_time=0.4)
        r_rel = self.rowt("RELEASE", 0, 28)
        r_nc = self.rowt("NO CORRECTION", 1, 28)
        self.at(C["release_word"])
        self.show(r_rel)
        self.at(C["go"])
        G1 = self.start_flight(P_GUST, C["go"], 2.2)
        self.play(pred.animate.set_stroke(opacity=0.45), run_time=0.3)

        tg = C["go"] + 2.2 * GUST_IDX / (len(P_GUST) - 1)
        pg = P_GUST[GUST_IDX]
        gust_g = VGroup(*[Arrow([min(pg[0] + 1.5, 3.4), pg[1] + dy, 0], [min(pg[0] + 0.6, 2.5), pg[1] + dy, 0],
                                buff=0, color=CYAN, stroke_width=2.2, tip_length=0.14)
                          for dy in (-0.3, 0, 0.3)]).set_z_index(5)
        self.at(tg - 0.2)
        self.play(FadeIn(gust_g), run_time=0.3)
        self.at(C["no_corr"])
        self.play(FadeIn(r_nc), FadeOut(gust_g), run_time=0.4)
        self.at(C["go"] + 2.2)
        gap2 = ground_cote(P_GUST[-1][0], XT)
        self.play(FadeIn(gap2), run_time=0.3)

        # ============================================================ 11 - IMPACT = f(RELEASE)
        self.at(C["eq_clear"])
        self.glide(XA0, C["eq_clear"] + 0.6, C["eq_clear"] + 0.601, smooth=False)
        self.play(FadeOut(VGroup(r_rel, r_nc, gap2)), FadeOut(G1.trail), FadeOut(pred), FadeOut(tgt),
                  FadeOut(bomb), FadeOut(air), run_time=0.3)
        self.fl = None
        toks = [TX(s, 34, INK) for s in ["IMPACT", "=", "f(", "RELEASE", ")"]]
        toks[3].set_color(AMBER)
        eq = VGroup(*toks).arrange(RIGHT, buff=0.22).move_to([0, 1.4, 0]).set_z_index(9)
        if eq.width > config.frame_width - 0.8:
            eq.scale((config.frame_width - 0.8) / eq.width)
        for cue, tk in ((C["eq1"], toks[0]), (C["eq2"], toks[1]), (C["eq3"], toks[2]),
                        (C["eq4"], toks[3]), (C["eq5"], toks[4])):
            self.at(cue)
            self.show(tk, rt=0.25)
        box = SurroundingRectangle(eq, color=AMBER, buff=0.25, stroke_width=2.5)
        self.at(C["eq_box"])
        self.play(Create(box), run_time=0.4)
        nxt = TX("", 30, CYAN).move_to([0, -0.4, 0])
        self.at(C["nxt"])
        self.show(nxt, rt=0.5)

        # raccord de boucle : retour a l'image 0
        self.at(C["loop"])
        air.update(0)
        bomb.update(0)
        self.play(FadeOut(eq), FadeOut(box), FadeOut(nxt), FadeIn(air), FadeIn(bomb),
                  FadeIn(hook_group), run_time=0.8)
        self.at(END_T)

        if self.late:
            print("SYNC WARNINGS - cue / lateness (seconds):")
            for cue, delay in self.late:
                print(f"  {cue:6.2f}s  +{delay:.3f}s")
            print(f"Total late cues: {len(self.late)} | max lateness: {max(d for _, d in self.late):.3f}s")
        else:
            print("SYNC CHECK: no cue overruns larger than one frame were detected.")