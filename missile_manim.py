from manim import *
import numpy as np
import os

PREVIEW = os.environ.get("PREVIEW") == "1"
PW = 540 if PREVIEW else 1080
config.pixel_width = PW
config.pixel_height = PW * 16 // 9
config.frame_width = 9.0
config.frame_height = 16.0
config.frame_rate = 15 if PREVIEW else 30  # keep 30 fps for final audio sync

# ------------------------------------------------------------------ style
BG = "#0b2a5b"      # bleu blueprint fonce
INK = "#f4f8ff"     # blanc
CYAN = "#8fe3ff"    # accent
AMBER = "#ffcf5a"   # accent final (erreur)
GRIDC = "#5b83bd"   # quieter grid keeps foreground readable
FONT = "DejaVu Sans Mono"
config.background_color = BG

# Mots affiches (change ici pour passer en francais)
L = dict(IS="IS", ISNT="ISN'T", WAS="WAS", WASNT="WASN'T", SHOULD="SHOULD BE", SHOULDNT="SHOULDN'T",
         DEV="DEVIATION", VAR="VARIATION", ERR="ERROR", SET="SETPOINT", MEAS="MEASURE",
         GUID="GUIDANCE", CMD="COMMAND", MIS="MISSILE")

# ------------------------------------------------------------------ geometrie
A_Y, B_Y = -2.5, 1.2       # position de depart / cible sur l'axe vertical
SC = 1.4                   # echelle du missile
GAP_X = -2.4               # x de la cote (deviation / variation)
END_T = 102.5              # fin de la video : garder une courte pause apres la narration
EQ_X0 = -3.7               # alignement gauche des equations

KEYS = [(0, A_Y), (25.4, A_Y), (29.3, B_Y), (41.7, B_Y), (43.4, B_Y - 0.55),
        (62.4, B_Y - 0.55), (63.5, B_Y), (9999, B_Y)]


NUM_CACHE = {}


def traj(t):
    for (t0, y0), (t1, y1) in zip(KEYS, KEYS[1:]):
        if t0 <= t < t1:
            u = (t - t0) / (t1 - t0)
            return y0 + (y1 - y0) * u * u * (3 - 2 * u)
    return KEYS[-1][1]


def TX(s, size=26, color=INK):
    return Text(s, font=FONT, font_size=size, color=color)


def block(label, w, h, size=22, color=INK):
    """Consistent block with a label that never spills outside its border."""
    r = RoundedRectangle(
        corner_radius=0.09, width=w, height=h,
        stroke_color=color, stroke_width=1.8,
        fill_color=BG, fill_opacity=1,
    )
    label_mob = TX(label, size, color)
    max_w, max_h = w - 0.22, h - 0.12
    if label_mob.width > max_w or label_mob.height > max_h:
        label_mob.scale(min(max_w / label_mob.width, max_h / label_mob.height))
    label_mob.move_to(r.get_center())
    return VGroup(r, label_mob)


def row(tokens, y, size=28, x0=EQ_X0, buff=0.3, color=INK, center=False):
    """Build a row and shrink it gently if it would exceed the 9:16 safe area."""
    toks = [TX(s, size, color) for s in tokens]
    g = VGroup(*toks).arrange(RIGHT, buff=buff)
    max_w = config.frame_width - 0.8  # 0.4-unit safe margin on both sides
    if g.width > max_w:
        g.scale(max_w / g.width)
    if center:
        g.move_to([0, y, 0])
    else:
        left_x = max(x0, -config.frame_width / 2 + 0.4)
        if left_x + g.width > config.frame_width / 2 - 0.4:
            left_x = config.frame_width / 2 - 0.4 - g.width
        g.move_to([left_x, y, 0], aligned_edge=LEFT)
    return toks, g


def make_missile():
    R = [(0, 1.0), (0.04, 0.93), (0.08, 0.82), (0.11, 0.68), (0.125, 0.52), (0.125, -0.04),
         (0.56, -0.40), (0.56, -0.50), (0.125, -0.50), (0.125, -0.80), (0.33, -0.93),
         (0.33, -1.0), (0.125, -0.97), (0.10, -1.0)]
    pts = R + [(-x, y) for x, y in reversed(R[1:])]
    poly = Polygon(*[[x, y, 0] for x, y in pts], stroke_color=INK, stroke_width=2.6,
                   fill_color=BG, fill_opacity=1)
    det = VGroup(
        Line([-0.11, 0.68, 0], [0.11, 0.68, 0]),
        Line([-0.125, 0.30, 0], [0.125, 0.30, 0]),
        Line([-0.125, -0.04, 0], [0.125, -0.04, 0]),
        Line([-0.125, -0.50, 0], [0.125, -0.50, 0]),
        Line([-0.125, -0.80, 0], [0.125, -0.80, 0]),
        Line([0.30, -0.20, 0], [0.30, -0.46, 0]),
        Line([-0.30, -0.20, 0], [-0.30, -0.46, 0]),
        Line([0, 0.68, 0], [0, -0.80, 0]),
    ).set_stroke(INK, 1.4, opacity=0.75)
    return VGroup(poly, det).scale(SC)


class Missile(Scene):
    def at(self, t):
        """Wait for an absolute audio cue and log only genuine overruns."""
        d = t - self.time
        frame = 1.0 / config.frame_rate
        if d > frame:
            self.wait(d)
        elif d < -frame:
            self.late.append((round(t, 2), round(-d, 3)))

    def show(self, *mobs, rt=0.35):
        self.play(*[FadeIn(m) for m in mobs], run_time=rt)

    def construct(self):
        self.late = []
        self.gap_mode = "D"
        self.m_y = ValueTracker(A_Y)
        self.gap_a = ValueTracker(0)
        self.gap_l = ValueTracker(0)
        self.ro_a = ValueTracker(0)   # opacite du compteur 0-100 a cote de IS
        self.m_y.add_updater(lambda m, dt: m.set_value(traj(self.time)))
        self.add(self.m_y)

        # ---------------------------------------------------- fond blueprint
        grid = VGroup()
        for i in range(-9, 10):
            maj = i % 4 == 0
            grid.add(Line(
                [i * 0.5, -8, 0], [i * 0.5, 8, 0], color=GRIDC,
                stroke_width=1.0 if maj else 0.55,
                stroke_opacity=0.22 if maj else 0.075,
            ))
        for j in range(-16, 17):
            maj = j % 4 == 0
            grid.add(Line(
                [-4.5, j * 0.5, 0], [4.5, j * 0.5, 0], color=GRIDC,
                stroke_width=1.0 if maj else 0.55,
                stroke_opacity=0.22 if maj else 0.075,
            ))
        border = Rectangle(
            width=8.5, height=15.5, color=INK,
            stroke_width=1.2, stroke_opacity=0.42,
        )
        grid.set_z_index(0)
        border.set_z_index(0)

        axis = DashedLine(
            [0, -3.95, 0], [0, 3.2, 0], dash_length=0.18,
            color=INK, stroke_width=1.2, stroke_opacity=0.38,
        ).set_z_index(1)

        # ---------------------------------------------------- missile
        miss = make_missile().move_to([0, A_Y, 0]).set_z_index(4)
        miss.add_updater(lambda m, dt: m.move_to([0, self.m_y.get_value(), 0]))

        flame = Polygon([0.12, 0, 0], [0, -0.7, 0], [-0.12, 0, 0], stroke_color=CYAN,
                        stroke_width=2.2, fill_opacity=0).set_z_index(3)
        flame_in = Polygon([0.05, 0, 0], [0, -0.38, 0], [-0.05, 0, 0], stroke_color=INK,
                           stroke_width=1.6, fill_opacity=0).set_z_index(3)
        flame_g = VGroup(flame, flame_in)

        def flame_up(m, dt):
            t = self.time
            v = traj(t + 0.15) - traj(t)
            op = 1.0 if v > 0.03 else 0.0
            fl = 1 + 0.22 * np.sin(t * 40)
            m.move_to([0, self.m_y.get_value() - SC * 1.0 - 0.36 * fl, 0])
            m.set_stroke(opacity=op)
        flame_g.add_updater(flame_up)

        # ---------------------------------------------------- valeurs numeriques 0 -> 100
        def numtext(v, size=32, color=CYAN):
            key = (v, size)
            if key not in NUM_CACHE:
                NUM_CACHE[key] = TX(str(v), size, color)
            return NUM_CACHE[key].copy()

        def pct(y):
            return int(round((y - A_Y) / (B_Y - A_Y) * 100))

        def readout_up(m, dt):
            y = self.m_y.get_value()
            m.become(numtext(pct(y), 32).move_to([2.35, y, 0], aligned_edge=LEFT)
                     .set_opacity(self.ro_a.get_value()))
        readout = VGroup()
        readout.add_updater(readout_up)

        # ---------------------------------------------------- tag "IS"
        dot = Dot([0, 0, 0], radius=0.075, color=INK)
        lead = Line([0.1, 0, 0], [1.4, 0, 0], stroke_width=1.6, color=INK)
        t_is = TX(L["IS"], 32).next_to(lead, RIGHT, buff=0.1)
        q = TX("?", 32, CYAN).next_to(t_is, RIGHT, buff=0.08)
        q.set_opacity(0)
        tag = VGroup(dot, lead, t_is, q).set_z_index(6)
        tag.shift(UP * A_Y)
        tag.add_updater(lambda m, dt: m.shift(UP * (self.m_y.get_value() - m[0].get_center()[1])))

        # nuage d'incertitude
        rng = np.random.default_rng(7)
        base = [(rng.normal(0, 0.16), rng.normal(0, 0.28), rng.uniform(0, 6.28), rng.uniform(0, 6.28))
                for _ in range(22)]
        cloud = VGroup(*[Dot(radius=0.04, color=CYAN) for _ in base]).set_z_index(7)

        def cloud_up(m, dt):
            t = self.time
            y = self.m_y.get_value()
            for d, (bx, by, p1, p2) in zip(m.submobjects, base):
                d.move_to([bx + 0.07 * np.sin(4 * t + p1), y + by + 0.07 * np.cos(5 * t + p2), 0])
        cloud.add_updater(cloud_up)

        # ---------------------------------------------------- cote (deviation / variation)
        lbl_D = TX("Δ", 36, CYAN)
        lbl_d = TX("δ", 36, CYAN)

        def build_gap():
            y1, y2 = self.m_y.get_value(), B_Y
            a = self.gap_a.get_value()
            if abs(y1 - y2) < 0.12 or a < 0.01:
                return VGroup()
            top, bot = max(y1, y2), min(y1, y2)
            ex = VGroup(*[DashedLine([-0.78, y, 0], [GAP_X - 0.3, y, 0], dash_length=0.08,
                                     stroke_width=1.2, color=CYAN) for y in (y1, y2)])
            ar = DoubleArrow([GAP_X, bot, 0], [GAP_X, top, 0], buff=0, stroke_width=2.4,
                             tip_length=0.16, color=CYAN, max_tip_length_to_length_ratio=0.45)
            g = VGroup(ex, ar).set_opacity(a)
            mid = (y1 + y2) / 2
            lb = (lbl_D if self.gap_mode == "D" else lbl_d).copy()
            lb.move_to([GAP_X - 0.5, mid + 0.27, 0]).set_opacity(self.gap_l.get_value())
            nm = numtext(int(round(abs(y1 - y2) / (B_Y - A_Y) * 100)), 30)
            nm.move_to([GAP_X - 0.5, mid - 0.27, 0]).set_opacity(self.gap_l.get_value())
            return VGroup(g, lb, nm)
        gap = VGroup()
        gap.add_updater(lambda m, dt: m.become(build_gap()))

        # ---------------------------------------------------- cible + reperes
        def datum(y, op=0.48):
            return DashedLine(
                [-3.9, y, 0], [3.9, y, 0], dash_length=0.12,
                color=INK, stroke_width=1.7, stroke_opacity=op,
            ).set_z_index(2)

        def dlabel(txt, y, color=INK, size=32):
            t = TX(txt, size, color)
            t.move_to([3.2, y + 0.42, 0], aligned_edge=RIGHT)
            return t.set_z_index(7)

        ring_B = VGroup(Circle(radius=0.26, color=INK, stroke_width=2.5),
                        *[Line(d * 0.18, d * 0.48, color=INK, stroke_width=2.2)
                          for d in (UP, DOWN, LEFT, RIGHT)]).move_to([0, B_Y, 0]).set_z_index(6)
        datum_B = datum(B_Y)
        lab_isnt = dlabel(L["ISNT"], B_Y)
        lab_wasnt = dlabel(L["WASNT"], B_Y)

        datum_A = datum(A_Y)
        ghost_poly = make_missile()[0].copy().set_fill(opacity=0)
        ghost = DashedVMobject(ghost_poly, num_dashes=110, dashed_ratio=0.55).set_stroke(INK, 2, opacity=0.6)
        ghost.move_to([0, A_Y, 0]).set_z_index(3)
        lab_A_was = dlabel(L["WAS"], A_Y)
        lab_A_isnt = dlabel(L["ISNT"], A_Y)
        lab_A_was2 = dlabel(L["WAS"], A_Y)

        # tolerance (+/- epsilon)
        tol = VGroup(*[DashedLine([1.95, B_Y + s * 0.25, 0], [2.95, B_Y + s * 0.25, 0], dash_length=0.07,
                                  color=CYAN, stroke_width=1.6) for s in (1, -1)])
        tol_lbl = TX("±ε", 28, CYAN).move_to([2.55, B_Y - 0.62, 0])

        # ---------------------------------------------------- boucle de regulation (haut)
        LY = 5.0
        b_sp = block(L["SET"], 1.95, 0.76, 26).move_to([-3.03, LY, 0])
        sig = VGroup(Circle(radius=0.25, stroke_color=INK, stroke_width=2, fill_color=BG, fill_opacity=1),
                     TX("Σ", 26)).move_to([-1.45, LY, 0])
        b_gd = block(L["GUID"], 1.95, 0.76, 26).move_to([0.175, LY, 0])
        b_ms = block(L["MIS"], 1.7, 0.76, 26).move_to([2.4, LY, 0])

        def arr(p, q_):
            return Arrow(p, q_, buff=0, stroke_width=2.4, tip_length=0.15, color=INK,
                         max_tip_length_to_length_ratio=0.6)
        loop_arrows = VGroup(
            arr([-2.1, LY, 0], [-1.72, LY, 0]),
            arr([-1.18, LY, 0], [-0.82, LY, 0]),
            arr([1.12, LY, 0], [1.58, LY, 0]),
            Line([3.2, LY, 0], [3.6, LY, 0], stroke_width=2.4, color=INK),
            Line([3.6, LY, 0], [3.6, 4.4, 0], stroke_width=2.4, color=INK),
            Line([3.6, 4.4, 0], [-1.45, 4.4, 0], stroke_width=2.4, color=INK),
            arr([-1.45, 4.4, 0], [-1.45, LY - 0.25, 0]),
        )
        lab_err = TX(L["ERR"], 24, AMBER).move_to([-1.0, LY + 0.6, 0])
        lab_meas = TX(L["MEAS"], 24, CYAN).move_to([1.1, 4.12, 0])
        pm = VGroup(TX("+", 22).move_to([-1.85, LY + 0.3, 0]), TX("−", 22).move_to([-1.12, 4.58, 0]))
        loop_all = VGroup(b_sp, sig, b_gd, b_ms, loop_arrows, lab_err, lab_meas, pm)

        # bruit sur la mesure
        nrng = np.random.default_rng(3)
        xs = np.linspace(3.6, -1.45, 70)
        npts = [[x, 4.4 + (nrng.normal(0, 0.11) if 3.0 > x > -1.1 else 0), 0] for x in xs]
        noise = VMobject(stroke_color=CYAN, stroke_width=2.6).set_points_as_corners(npts)

        sweep = Line([-4.25, 0, 0], [4.25, 0, 0], color=CYAN, stroke_width=1.6, stroke_opacity=0.10).set_z_index(1)
        sweep.add_updater(lambda m, dt: m.move_to([0, 8 - 16 * ((self.time / (END_T / 8)) % 1), 0]))

        # ======================================================= TIMELINE
        # --- Hook immediat, place sous la zone d'interface de Reels (y < 5.9)
        hook = TX("THE MISSILE KNOWS", 44, INK).move_to([0, 5.2, 0])
        hook_rule = Line([-3.0, 4.72, 0], [3.0, 4.72, 0], color=CYAN, stroke_width=2.0)
        hook_group = VGroup(hook, hook_rule).set_z_index(10)
        hook2 = TX("WHERE IT IS", 44, CYAN).move_to([0, 4.2, 0]).set_z_index(10)
        self.add(hook_group)
        self.add(gap, flame_g, readout)
        readout.set_z_index(6)
        gap.set_z_index(5)
        # Frame 0 : missile + titre deja visibles. La cible n'apparait qu'a 7.1 s
        # pour que "where it isn't" soit bien une revelation.
        self.add(grid, border, axis, sweep, miss)   # image 0 = image finale (raccord de boucle)
        self.at(2.239)
        # 2.24 "The missile knows where it is at all times"
        self.play(Indicate(miss, scale_factor=1.07, color=CYAN), run_time=0.6)
        self.at(3.42)
        self.play(FadeIn(tag), FadeIn(hook2), self.ro_a.animate.set_value(1), run_time=0.55)
        self.at(4.08)
        self.play(Indicate(tag[0], scale_factor=2.4, color=CYAN), run_time=0.5)
        self.at(4.7)
        self.play(FadeOut(hook_group), FadeOut(hook2), run_time=0.4)

        # 5.74 "It knows this because it knows where it isn't"
        self.at(5.74)
        scan = Circle(radius=0.3, color=CYAN, stroke_width=2.2).move_to([0, A_Y, 0]).set_z_index(3)
        self.add(scan)
        self.play(scan.animate.scale(10).set_stroke(opacity=0), run_time=1.3, rate_func=linear)
        self.remove(scan)
        self.at(7.1)
        self.play(Create(datum_B), FadeIn(ring_B, scale=1.6), run_time=0.4)
        self.at(7.5)
        self.play(FadeIn(lab_isnt), run_time=0.4)

        # 8.58 "By subtracting where it is from where it isn't ..."
        self.at(8.58)
        self.play(self.gap_a.animate.set_value(1), run_time=0.5)
        e1, g1 = row([L["ISNT"], "−", L["IS"], "= Δ"], 4.5, 35)
        e2, g2 = row([L["IS"], "−", L["ISNT"], "= −Δ"], 3.8, 35)
        self.at(9.66)
        self.play(FadeIn(e1[2]), Indicate(tag[0], scale_factor=2.2, color=CYAN), run_time=0.45)
        self.at(10.3)
        self.show(e1[1], rt=0.3)
        self.at(10.62)
        self.play(FadeIn(e1[0]), Indicate(lab_isnt, color=CYAN), run_time=0.5)
        self.at(12.24)
        self.show(e2[2], rt=0.35)
        self.at(12.7)
        self.show(e2[1], rt=0.3)
        self.at(13.06)
        self.show(e2[0], rt=0.35)
        self.at(13.7)
        self.show(e1[3], e2[3], rt=0.4)
        self.at(14.26)
        self.play(VGroup(*e1).animate.set_color(CYAN), VGroup(*e2).animate.set_opacity(0.25), run_time=0.5)
        self.at(15.2)
        self.play(FadeOut(VGroup(*e2)), run_time=0.4)
        self.at(16.04)
        self.play(self.gap_l.animate.set_value(1), run_time=0.4)
        kw_dev = TX(L["DEV"], 22, CYAN).move_to([EQ_X0, 3.85, 0], aligned_edge=LEFT)
        self.at(16.8)
        self.show(kw_dev, rt=0.5)
        self.play(Indicate(kw_dev, color=CYAN, scale_factor=1.08), run_time=0.35)

        # 18.4 "The guidance subsystem uses deviations to generate corrective commands ..."
        self.at(18.4)
        self.play(FadeOut(VGroup(*e1)), FadeOut(kw_dev), run_time=0.3)
        CY = 4.7
        c_delta = TX("Δ", 36, CYAN).move_to([-3.2, CY, 0])
        c_blk = block(L["GUID"], 2.5, 0.7, 24).move_to([0, CY, 0])
        c_cmd = TX(L["CMD"], 24).move_to([3.05, CY, 0])
        c_a1 = arr([-2.85, CY, 0], [-1.3, CY, 0])
        c_a2 = arr([1.3, CY, 0], [2.3, CY, 0])
        self.at(18.74)
        self.show(c_blk, rt=0.5)
        self.at(20.3)
        self.show(c_delta, c_a1, rt=0.5)
        self.at(21.86)
        self.show(c_a2, rt=0.4)
        self.at(22.5)
        self.show(c_cmd, rt=0.4)
        self.at(23.4)
        self.play(Indicate(c_cmd, color=CYAN), run_time=0.6)
        self.at(24.85)
        self.play(Indicate(tag[0], scale_factor=2.4, color=CYAN), run_time=0.5)    # le vol demarre a 25.4
        self.at(26.12)
        self.play(Indicate(ring_B, scale_factor=1.5, color=CYAN), run_time=0.8)
        self.at(27.14)
        self.play(Indicate(lab_isnt, color=CYAN), run_time=0.5)
        self.at(29.74)
        self.play(ReplacementTransform(lab_isnt, lab_wasnt), run_time=0.5)
        self.at(30.38)
        self.play(Indicate(tag[0], scale_factor=2.4, color=CYAN),
                  FadeOut(VGroup(c_delta, c_blk, c_cmd, c_a1, c_a2)), run_time=0.5)

        # 31.92 "Consequently, the position where it is is now the position that it wasn't ..."
        e3, g3 = row([L["IS"], "=", L["WASNT"]], 3.8, 35)
        e4, g4 = row([L["WAS"], "=", L["ISNT"]], 3.15, 35)
        self.at(33.0)
        self.play(FadeIn(e3[0]), Indicate(tag[0], scale_factor=2.2, color=CYAN), run_time=0.45)
        self.at(34.76)
        self.show(e3[1], rt=0.3)
        self.at(35.5)
        self.play(FadeIn(e3[2]), Indicate(lab_wasnt, color=CYAN), run_time=0.5)
        # "...and it follows that the position that it was is now the position that it isn't"
        self.at(37.7)
        self.play(FadeIn(ghost), Create(datum_A), FadeIn(lab_A_was), run_time=0.6)
        self.at(38.4)
        self.show(e4[0], rt=0.3)
        self.at(39.4)
        self.show(e4[1], rt=0.3)
        self.at(39.84)
        self.play(FadeIn(e4[2]), ReplacementTransform(lab_A_was, lab_A_isnt), run_time=0.5)
        self.at(41.0)
        self.play(FadeOut(VGroup(*e3)), FadeOut(VGroup(*e4)), self.gap_l.animate.set_value(0), run_time=0.5)
        self.gap_mode = "d"

        # 43.72 "In the event that the position that it is in is not the position that it wasn't ..."
        e5, g5 = row([L["IS"], "≠", L["WASNT"]], 3.8, 35)
        self.at(43.72)  # reprise de la voix apres le silence : petit pulse sur le missile
        self.play(Indicate(miss, scale_factor=1.04, color=CYAN), run_time=0.5)
        self.at(45.8)   # "is" est prononce ici
        self.play(FadeIn(e5[0]), Indicate(tag[0], scale_factor=2.2, color=CYAN), run_time=0.4)
        self.at(46.5)
        self.show(e5[1], rt=0.3)
        self.at(47.4)
        self.play(FadeIn(e5[2]), Indicate(lab_wasnt, color=CYAN), run_time=0.45)
        # "...the system has acquired a variation"
        self.at(49.1)
        self.play(FadeOut(VGroup(*e5)), run_time=0.3)
        kw_var = TX(L["VAR"], 26, CYAN).move_to([EQ_X0, 3.8, 0], aligned_edge=LEFT)
        self.at(50.3)
        self.play(FadeIn(kw_var), self.gap_l.animate.set_value(1), run_time=0.5)
        self.play(Indicate(kw_var, color=CYAN, scale_factor=1.08), run_time=0.35)
        # "...the variation being the difference between where the missile is and where it wasn't"
        e6, g6 = row(["δ", "=", L["IS"], "−", L["WASNT"]], 3.15, 35)
        self.at(51.9)
        self.show(e6[0], rt=0.3)
        self.at(52.5)
        self.show(e6[1], rt=0.3)
        self.at(54.0)
        self.play(FadeIn(e6[2]), Indicate(tag[0], scale_factor=2.2, color=CYAN), run_time=0.4)
        self.at(54.9)
        self.show(e6[3], rt=0.25)
        self.at(55.3)
        self.play(FadeIn(e6[4]), Indicate(lab_wasnt, color=CYAN), run_time=0.45)
        # 56.9 "If variation is considered to be a significant factor ..."
        e7, g7 = row(["δ", ">", "ε", "✓"], 3.15, 35)
        self.at(57.0)
        self.play(FadeOut(VGroup(*e6)), run_time=0.3)
        self.show(e7[0], rt=0.3)
        self.at(58.05)
        self.show(e7[1], rt=0.25)
        self.at(58.95)
        self.show(e7[2], rt=0.3)
        self.at(59.6)
        self.play(FadeIn(e7[3]), run_time=0.3)
        # 60.6 "it, too, may be corrected by the GEA"
        self.at(60.6)
        self.play(FadeOut(VGroup(*e7)), run_time=0.3)
        g_delta = TX("δ", 36, CYAN).move_to([-3.2, CY, 0])
        g_blk = block("GEA", 2.5, 0.7, 24).move_to([0, CY, 0])
        g_cmd = TX(L["CMD"], 24).move_to([3.05, CY, 0])
        g_a1 = arr([-2.85, CY, 0], [-1.3, CY, 0])
        g_a2 = arr([1.3, CY, 0], [2.3, CY, 0])
        self.at(60.95)
        self.show(g_delta, g_a1, rt=0.4)
        self.at(62.3)
        self.show(g_blk, g_a2, g_cmd, rt=0.5)
        self.at(63.6)
        self.play(FadeOut(VGroup(g_delta, g_blk, g_cmd, g_a1, g_a2)), FadeOut(kw_var), run_time=0.4)

        # 64.2 "However, the missile must also know where it was"
        self.at(65.5)
        self.play(ReplacementTransform(lab_A_isnt, lab_A_was2), ghost.animate.set_stroke(INK, 2.6, opacity=1),
                  run_time=0.5)
        self.play(Indicate(ghost, scale_factor=1.04, color=CYAN), run_time=0.7)

        # 67.44 "The missile guidance computer scenario works as follows"
        self.at(67.44)
        # Une seule animation : on attenue la scene ET on fait apparaitre MISSILE,
        # pour que "guidance" (68.0) tombe a l'heure.
        self.play(miss.animate.set_opacity(0.38), ghost.animate.set_opacity(0.28),
                  ring_B.animate.set_opacity(0.35), datum_A.animate.set_opacity(0.18),
                  datum_B.animate.set_opacity(0.18), FadeIn(b_ms), run_time=0.5)
        self.at(68.0)
        self.show(b_gd, rt=0.5)
        self.at(69.0)
        self.show(b_sp, sig, pm, rt=0.5)
        self.at(69.6)
        self.play(Create(loop_arrows), run_time=0.9)
        self.at(70.5)
        self.show(lab_err, lab_meas, rt=0.4)
        self.play(Indicate(b_gd, color=CYAN, scale_factor=1.06), run_time=0.35)

        # 71.36 "Because a variation has modified some of the information the missile has obtained ..."
        self.at(72.0)
        self.play(Create(noise), run_time=2.0, rate_func=linear)
        # "...it is not sure just where it is"
        self.at(76.5)
        self.play(tag[0].animate.set_opacity(0), FadeIn(cloud), self.ro_a.animate.set_value(0), run_time=0.5)
        self.at(77.5)
        self.play(q.animate.set_opacity(1), run_time=0.4)
        # 79.1 "However, it is sure where it isn't, within reason ..."
        self.at(80.2)
        self.play(Indicate(ring_B, scale_factor=1.5, color=CYAN), FadeOut(noise), run_time=0.8)
        self.at(81.5)
        self.show(tol, tol_lbl, rt=0.6)
        # "...and it knows where it was"
        self.at(82.7)
        self.play(miss.animate.set_opacity(1), ghost.animate.set_opacity(0.6),
                  ring_B.animate.set_opacity(1), datum_A.animate.set_opacity(0.48),
                  datum_B.animate.set_opacity(0.48), run_time=0.35)
        self.play(Indicate(ghost, scale_factor=1.06, color=CYAN),
                  Indicate(lab_A_was2, color=CYAN), run_time=0.8)

        # 84.4 "It now subtracts where it should be from where it wasn't, or vice versa"
        rA, gA = row(["(", L["WASNT"], "−", L["SHOULD"], ")"], 3.7, 30)
        rB, gB = row(["−", "(", L["SHOULDNT"], "+", L["WAS"], ")"], 3.0, 30)
        rA[1].set_color(CYAN)
        rB[4].set_color(CYAN)
        self.at(85.0)
        self.show(rA[2], rt=0.3)
        self.at(85.7)
        self.show(rA[3], rt=0.35)
        self.at(86.9)
        self.play(FadeIn(rA[1]), Indicate(lab_wasnt, color=CYAN), run_time=0.45)
        self.at(87.6)
        tmp, _ = row(["(", L["SHOULD"], "−", L["WASNT"], ")"], 3.7, 30)
        self.play(rA[3].animate.move_to(tmp[1].get_center()),
                  rA[2].animate.move_to(tmp[2].get_center()),
                  rA[1].animate.move_to(tmp[3].get_center()), run_time=0.8)
        # 89.0 "And by differentiating this from the algebraic sum of where it shouldn't be and where it was"
        self.at(89.2)
        self.show(rA[0], rA[4], rt=0.3)
        self.at(90.6)
        self.show(rB[0], rt=0.3)
        self.at(91.2)
        self.show(rB[1], rB[5], rt=0.3)
        self.at(92.8)
        self.show(rB[2], rt=0.35)
        self.at(93.4)
        self.show(rB[3], rt=0.25)
        self.at(94.1)
        self.play(FadeIn(rB[4]), Indicate(ghost, scale_factor=1.04, color=CYAN), run_time=0.45)

        # 94.98 "it is able to obtain the deviation and its variation, which is called error"
        # La ligne resultat est placee sous la boucle (y=3.6) et la boucle est attenuee.
        res, gres = row(["Δ", "+", "δ"], 4.4, 44, center=True, buff=0.9)
        res[0].set_color(CYAN)
        res[2].set_color(CYAN)
        k1 = TX(L["DEV"], 22, CYAN).next_to(res[0], DOWN, buff=0.15).shift(LEFT * 0.4)
        k2 = TX(L["VAR"], 22, CYAN).next_to(res[2], DOWN, buff=0.15).shift(RIGHT * 0.4)

        self.at(94.85)
        self.play(FadeOut(VGroup(*rA)), FadeOut(VGroup(*rB)), FadeOut(loop_all), run_time=0.4)
        self.at(96.36)
        self.play(FadeIn(res[0], scale=1.4), FadeIn(k1), run_time=0.4)
        self.at(96.9)
        self.show(res[1], rt=0.25)
        self.at(97.5)
        self.play(FadeIn(res[2], scale=1.4), FadeIn(k2), run_time=0.4)
        self.at(98.38)

        box = SurroundingRectangle(VGroup(gres, k1, k2), color=AMBER, buff=0.2, stroke_width=2.5)
        self.play(Create(box), run_time=0.55)
        self.at(99.05)

        # Climax : tout s'efface, ERROR devient l'unique point focal (une seule animation).
        err = TX(L["ERR"], 68, AMBER).move_to([0, 1.5, 0])
        self.play(FadeOut(VGroup(miss, flame_g, tag, cloud, ghost, ring_B, datum_A, datum_B,
                                 lab_A_was2, lab_wasnt, tol, tol_lbl, gap)),
                  self.ro_a.animate.set_value(0),
                  ReplacementTransform(VGroup(*res, box, k1, k2), err),
                  run_time=0.5)

        # fin : equation de la regulation
        self.at(99.7)
        fin = TX(f"{L['ERR']} = {L['SET']} − {L['MEAS']}", 30).move_to([0, 0.15, 0])
        if fin.width > config.frame_width - 0.8:
            fin.scale_to_fit_width(config.frame_width - 0.8)
        fin.set_color(INK)
        self.play(FadeIn(fin, shift=0.12 * UP), run_time=0.6)
        self.play(err.animate.scale(1.12), run_time=0.18, rate_func=there_and_back)
        self.play(Indicate(err, scale_factor=1.08, color=AMBER), run_time=0.7)
        # raccord de boucle : retour a l'image 0 (missile au depart + accroche)
        miss_end = make_missile().move_to([0, A_Y, 0]).set_z_index(4)
        self.at(101.7)
        self.play(FadeOut(err), FadeOut(fin), FadeIn(miss_end), FadeIn(hook_group), run_time=0.8)
        self.at(END_T)

        if self.late:
            print("SYNC WARNINGS — cue / lateness (seconds):")
            for cue, delay in self.late:
                print(f"  {cue:6.2f}s  +{delay:.3f}s")
            print(
                f"Total late cues: {len(self.late)} | "
                f"max lateness: {max(delay for _, delay in self.late):.3f}s"
            )
        else:
            print("SYNC CHECK: no cue overruns larger than one frame were detected.")
