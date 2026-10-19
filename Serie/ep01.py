"""
Serie "Types de guidage" - Episode 1 : l'absence de guidage
(bombes et roquettes non guidees, outils embarques de prediction du point d'impact : CCIP / CCRP).

Un seul fichier par episode : script de voix off + cues relatives + scene.
Tout le reste (synchro WhisperX, bruitages, mixage) est dans guidage.py :

    python3 guidage.py ep01.py script                        # texte a lire
    python3 guidage.py ep01.py build --preview               # apercu complet SANS voix (cues estimees)
    python3 guidage.py ep01.py build --voice voix.wav        # rendu final synchronise sur ta voix
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidage import *  # noqa: E402,F401,F403

# =============================================================================================
# VOIX OFF
#   [[cle]]mot       l'image "cle" se declenche au debut du mot qui suit
#   [[cle@end]]mot   ... a la fin du mot      [[cle+0.2]] / [[cle@end-0.1]] : decalage en secondes
#   {pause 1 s}      silence a laisser a la lecture (utile a l'animation) ; ignore par l'analyse
#   lignes # : commentaires
# =============================================================================================
SCRIPT = """
The bomb knows [[hook2]]nothing.

No [[no_comp]]computer. No [[no_sens]]sensor. No [[no_steer]]steering. Only [[physics]]physics.

[[release-0.15]][[g_arrow+0.1]]Gravity pulls it down, its [[v_arrow]]speed carries it forward: a [[ballistic]]ballistic [[land@end]]curve. {pause 1 s}

[[drag]]Drag slows it down. [[wind]]Wind pushes it off course. The [[error]]error is set at [[grows]]release.

So the intelligence must be in the [[aircraft]]aircraft.

A [[computer]]computer measures [[in_speed]]speed, [[in_alt]]altitude and [[in_wind]]wind, and [[solves]]solves the trajectory.

The [[ccip]]CCIP [[impact_pt]]shows where the bomb would [[shows]]land if released [[now]]now.

The pilot [[move]]moves the marker onto the [[target]]target, and [[rel_ccip]]releases. {pause 1.2 s}

The [[ccrp]]CCRP [[ccrp_flight]]reverses the problem: the pilot [[fly]]flies to the [[ccrp_target]]target, and the computer [[ccrp_mark]]calculates the [[ccrp_label]]release [[ccrp_rel@end]]point. {pause 1.2 s}

Unguided [[rocket]]rockets work the same way: no [[no_steer2]]steering, only a [[pred_rocket]]predicted impact [[fire@end]]point. {pause 1.6 s}

But [[prediction]]after [[release_word]][[go@end]]release, [[no_corr]]nothing can be corrected.

The [[eq1]]impact [[eq2]]is a [[eq3]]function of the [[eq4]][[eq5@end]]release.

[[nxt]]Next: guided [[loop@end+0.5]]flight.
"""

# Cues sans mot dedie : deduites d'une autre (laisser de la marge : l'outil mesure les ecarts minimaux)
RELATIVE = {
    "hook_out": "hook2+0.9",
    "curve": "ballistic-0.5",
    "air": "drag-0.55",
    "gap": "error-0.5",
    "to_air": "aircraft-1.9",
    "back": "ccip-1.0",
    "put": "rel_ccip-0.8",
    "eq_clear": "eq1-0.5",
    "eq_box": "eq5+0.4",
    "end": "loop+1.0",
}

# Aide a la transcription (mots rares)
WHISPER_PROMPT = ("CCIP, CCRP, ballistic trajectory, drag, wind, aircraft, unguided rockets, "
                  "impact point, release point.")

# =============================================================================================
# DECOR ET PHYSIQUE (illustratifs)
# =============================================================================================
AY, GY = 2.2, -3.7           # altitude de l'avion / sol
XA0 = -3.0                   # point de largage "naif"
START_DY = -0.30             # la bombe est sous l'avion
SX, G, DRAG_K, WIND = 2.8, 3.05, 0.05, -0.9
GUST = (0.7, (-2.4, 0.0))    # rafale apres largage : (instant, vent ajoute)
GROUND = GY + 0.05

P0 = (XA0, AY + START_DY)
P_VAC = simulate(P0, (SX, 0), g=G, ground=GROUND)                                    # gravite seule
P_DRAG = simulate(P0, (SX, 0), g=G, ground=GROUND, k=DRAG_K)                         # + trainee
P_WIND = simulate(P0, (SX, 0), g=G, ground=GROUND, k=DRAG_K, wind=(WIND, 0))         # + vent
XT = P_VAC[-1][0]                       # cible = impact "ideal" sans air
R_REAL = P_WIND[-1][0] - XA0            # portee reelle
XA1 = XT - R_REAL                       # bon point de largage
P_GUST = simulate((XA1, AY + START_DY), (SX, 0), g=G, ground=GROUND, k=DRAG_K,
                  wind=(WIND, 0), gust=GUST)
GUST_FRAC = int(GUST[0] / 0.008) / (len(P_GUST) - 1)


def solve_rocket(speed=9.0, k=0.01):
    """Angle de tir pour que la roquette touche XT."""
    lo, hi = -1.45, -0.05
    for _ in range(45):
        mid = (lo + hi) / 2
        end = simulate(P0, (speed * np.cos(mid), speed * np.sin(mid)), g=G, ground=GROUND, k=k)[-1][0]
        lo, hi = (mid, hi) if end < XT else (lo, mid)
    a = (lo + hi) / 2
    return simulate(P0, (speed * np.cos(a), speed * np.sin(a)), g=G, ground=GROUND, k=k)


P_ROCKET = solve_rocket()


# =============================================================================================
# SCENE
# =============================================================================================
class Episode(BlueprintScene):
    # ---- objets suivant l'avion / la trajectoire en cours
    def make_proj(self, shape):
        shape.set_z_index(6)
        shape.ang = 0.0

        def up(m, dt):
            if self.fl is None:
                p, ang = np.array([self.track.val(), AY + START_DY, 0.0]), 0.0
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
                ar, lb = vec(p + DOWN * 0.15, DOWN, 0.85), TX("g", 28, CYAN).move_to(p + np.array([0.3, -0.6, 0]))
            elif kind == "v":
                ar, lb = vec(p + d * 0.35, d, 1.0), TX("v", 28, CYAN).move_to(p + d * 1.5 + UP * 0.28)
            else:
                ar, lb = vec(p - d * 0.35, -d, 0.8), TX("D", 28, CYAN).move_to(p - d * 1.45 + UP * 0.28)
            m.become(VGroup(ar, lb).set_opacity(a))
            m.set_z_index(7)
        g.add_updater(up)
        return g

    def make_pred(self, x):
        pts = shift_path(P_WIND, x, AY + START_DY)
        pts = np.vstack([pts[::4], pts[-1:]])
        d = DashedVMobject(VMobject().set_points_as_corners(pts), num_dashes=46,
                           dashed_ratio=0.55).set_stroke(CYAN, 2.0).set_z_index(2)
        d.last_x = x

        def up(m, dt):
            nx = self.track.val()
            m.shift(RIGHT * (nx - m.last_x))
            m.last_x = nx
        d.add_updater(up)
        return d

    def pipper(self, follow=True):
        p = VGroup(Circle(radius=0.15, color=CYAN, stroke_width=2.2),
                   Dot(radius=0.035, color=CYAN)).set_z_index(6).move_to([XA0 + R_REAL, GY + 0.5, 0])
        if follow:
            p.add_updater(lambda m, dt: m.move_to([self.track.val() + R_REAL, GY + 0.5, 0]))
        return p

    # ---- timeline
    def construct(self):
        self.fl = None
        track = self.track = Track(self, XA0)
        ga, va, da = ValueTracker(0), ValueTracker(0), ValueTracker(0)

        grid, border, sweep = self.background()
        ground = VGroup(
            Line([-4.25, GY, 0], [4.25, GY, 0], color=INK, stroke_width=2.2),
            *[Line([x, GY, 0], [x - 0.2, GY - 0.2, 0], color=INK, stroke_width=1.0, stroke_opacity=0.45)
              for x in np.arange(-4.0, 4.3, 0.35)]).set_z_index(1)
        air = aircraft_shape().set_z_index(5).move_to([XA0, AY, 0])
        air.add_updater(lambda m, dt: m.move_to([track.val(), AY, 0]))
        bomb = self.make_proj(bomb_shape())
        rocket = self.make_proj(rocket_shape())
        g_g, v_g, d_g = self.make_follow("g", ga), self.make_follow("v", va), self.make_follow("d", da)
        hook_group, hook2 = make_hook("THE BOMB KNOWS", "NOTHING")

        self.add(grid, border, sweep, ground, air, bomb, hook_group, g_g, v_g, d_g)

        # ---------------------------------------------------------------- accroche
        self.at("hook2")
        self.play(FadeIn(hook2), run_time=0.4)
        self.at("hook_out")
        self.play(FadeOut(hook_group), FadeOut(hook2), run_time=0.3)

        # ---------------------------------------------------------------- pas de guidage
        r1, r2, r3 = self.rowt("NO COMPUTER", 0, 26), self.rowt("NO SENSOR", 1, 26), self.rowt("NO STEERING", 2, 26)
        for key, row in (("no_comp", r1), ("no_sens", r2), ("no_steer", r3)):
            self.at(key)
            self.play(FadeIn(row), self.ping(bomb.get_center()), run_time=0.4, sfx="tick")
        ph = TX("PHYSICS", 30, CYAN).move_to([EQ_X0, ROW_Y0, 0], aligned_edge=LEFT).set_z_index(9)
        self.at("physics")
        self.play(FadeOut(VGroup(r1, r2, r3)), FadeIn(ph), run_time=0.4)

        # ---------------------------------------------------------------- trajectoire balistique
        self.at("release")
        V = self.flight(P_VAC, "release", end="land", clack=True, gain=0.6)
        self.need("land", "release", 2.5)
        self.at("g_arrow")
        self.play(ga.animate.set_value(1), run_time=0.4, sfx="tick")
        self.at("v_arrow")
        self.play(va.animate.set_value(1), run_time=0.4, sfx="tick")
        self.at("curve")
        self.play(ga.animate.set_value(0), va.animate.set_value(0), run_time=0.4, sfx=False)
        r_bt = TX("BALLISTIC TRAJECTORY", 24, CYAN).move_to([EQ_X0, ROW_Y0 - ROW_DY, 0],
                                                            aligned_edge=LEFT).set_z_index(9)
        self.at("ballistic")
        self.show(r_bt)
        tgt = target_at(XT, GY)
        self.at("land")
        self.play(FadeIn(tgt, scale=1.4), run_time=0.4, sfx=False)
        self.play(FadeOut(VGroup(ph, r_bt)), FadeOut(bomb), V.trail.animate.set_stroke(opacity=0.3),
                  run_time=0.35)
        self.fl = None
        bomb.update(0)

        # ---------------------------------------------------------------- l'air devie la courbe
        self.at("air")
        self.play(FadeIn(bomb), run_time=0.3, sfx=False)
        r_drag, r_wind, r_err = self.rowt("DRAG", 0, 28), self.rowt("WIND", 1, 28), self.rowt("ERROR", 2, 28)
        self.at("drag")
        D = self.flight(P_DRAG, "drag", dur=1.0, f0=300, f1=1600, gain=0.5, thud=0.5)
        self.play(FadeIn(r_drag), da.animate.set_value(1), run_time=0.3)
        self.at("drag", 1.0)
        self.play(da.animate.set_value(0), FadeOut(bomb), D.trail.animate.set_stroke(opacity=0.3),
                  run_time=0.2, sfx=False)
        self.fl = None
        bomb.update(0)
        self.play(FadeIn(bomb), run_time=0.2, sfx=False)

        wind_g = VGroup(*[Arrow([3.1, y, 0], [1.7, y, 0], buff=0, color=CYAN, stroke_width=2.2, tip_length=0.14)
                          for y in (1.9, 1.5, 1.1)]).set_z_index(5)
        self.at("wind")
        W = self.flight(P_WIND, "wind", dur=1.2, f0=400, f1=2400, gain=0.6, thud=0.7)
        self.sfx("whoosh", t="wind", dur=1.2, f0=900, f1=3200, gain=0.15, pan=0.2)
        self.play(FadeIn(r_wind), FadeIn(wind_g), run_time=0.4)
        gap1 = ground_cote(P_WIND[-1][0], XT, GY)
        self.need("gap", "wind", 1.3)
        self.at("gap")
        self.play(FadeIn(gap1), run_time=0.4)
        self.at("error")
        self.play(FadeIn(r_err), run_time=0.35)
        self.at("grows")
        self.play(Indicate(gap1[2], color=CYAN, scale_factor=1.4), run_time=0.5)

        # ---------------------------------------------------------------- l'intelligence est dans l'avion
        self.at("to_air")
        self.play(FadeOut(VGroup(r_drag, r_wind, r_err)), FadeOut(wind_g), FadeOut(bomb), FadeOut(gap1),
                  FadeOut(V.trail), FadeOut(D.trail), FadeOut(W.trail), run_time=0.4)
        self.fl = None
        track.glide(XA1, ("to_air", 0.5), ("to_air", 1.2))
        self.at("to_air", 1.2)
        bomb.update(0)
        self.play(FadeIn(bomb), run_time=0.3, sfx=False)

        in_rows = [self.rowt(s, i, 22) for i, s in enumerate(["SPEED", "ALTITUDE", "WIND"])]
        cyc = ROW_Y0 - ROW_DY
        c_blk = block("COMPUTER", 2.2, 1.4, 22).move_to([0, cyc, 0]).set_z_index(9)
        in_arrows = [arr([-1.95, ROW_Y0 - i * ROW_DY, 0], [-1.15, ROW_Y0 - i * ROW_DY, 0]).set_z_index(9)
                     for i in range(3)]
        out_ar = arr([1.1, cyc, 0], [1.6, cyc, 0]).set_z_index(9)
        out_lb = fit_right(TX("TRAJECTORY", 22, CYAN).move_to([1.7, cyc, 0], aligned_edge=LEFT)).set_z_index(9)
        pred = self.make_pred(XA1)

        self.at("aircraft")
        self.play(self.ping(air.get_center()), run_time=0.5)
        self.at("computer")
        self.show(c_blk, rt=0.4)
        for key, i in (("in_speed", 0), ("in_alt", 1), ("in_wind", 2)):
            self.at(key)
            self.play(FadeIn(in_rows[i]), FadeIn(in_arrows[i]), run_time=0.35)
        self.at("solves")
        self.play(FadeIn(out_ar), FadeIn(out_lb), Create(pred), run_time=0.8)

        # ---------------------------------------------------------------- CCIP
        self.at("back")
        track.glide(XA0, "back", ("back", 0.9))
        self.play(FadeOut(VGroup(*in_rows, *in_arrows, c_blk, out_ar, out_lb)), run_time=0.5, sfx=False)
        rc = [self.rowt("CCIP", 0, 28), self.rowt("IF RELEASED NOW", 1, 24), self.rowt("TARGET", 2, 24)]
        pip = self.pipper()
        self.at("ccip")
        self.show(rc[0])
        self.at("impact_pt")
        self.play(FadeIn(pip), run_time=0.4)
        self.at("shows")
        self.play(self.ping(pip.get_center()), run_time=0.5)
        self.at("now")
        self.show(rc[1])
        self.at("move")
        track.glide(XA1, "move", "put")
        self.need("put", "move", 1.5)
        self.at("target")
        self.show(rc[2])
        self.at("put")
        self.play(Indicate(tgt, color=CYAN, scale_factor=1.15), run_time=0.5, sfx="lock")
        self.at("rel_ccip")
        F1 = self.flight(shift_path(P_WIND, XA1, AY + START_DY), "rel_ccip", dur=1.3, clack=True)
        self.play(FadeOut(pred), FadeOut(pip), run_time=0.3, sfx=False)
        self.at("rel_ccip", 1.3)
        track.glide(XA0, ("rel_ccip", 1.4), ("rel_ccip", 2.1))
        self.play(self.ping(tgt.get_center()), FadeOut(VGroup(*rc)), FadeOut(F1.trail), FadeOut(bomb),
                  run_time=0.5, sfx=False)
        self.fl = None

        # ---------------------------------------------------------------- CCRP
        rr = [self.rowt("CCRP", 0, 28), self.rowt("TARGET", 1, 24), self.rowt("RELEASE POINT", 2, 24)]
        fline = DashedLine([-4.0, AY, 0], [3.4, AY, 0], dash_length=0.12, color=INK, stroke_width=1.5,
                           stroke_opacity=0.4).set_z_index(2)
        rel_mark = VGroup(Line([XA1, AY - 0.65, 0], [XA1, AY + 0.65, 0], color=CYAN, stroke_width=2.0),
                          Triangle(color=CYAN, fill_color=CYAN, fill_opacity=1, stroke_width=1)
                          .scale(0.12).rotate(PI).move_to([XA1, AY + 0.82, 0])).set_z_index(3)
        self.at("ccrp")
        self.show(rr[0])
        self.at("ccrp", 0.6)
        bomb.update(0)
        self.play(FadeIn(bomb), run_time=0.3, sfx=False)
        self.at("ccrp_flight")
        self.show(fline, rt=0.4)
        self.at("fly")
        track.glide(XA1, "fly", "ccrp_rel", smooth=False)
        self.need("ccrp_rel", "fly", 2.0)
        self.at("ccrp_target")
        self.play(FadeIn(rr[1]), Indicate(tgt, color=CYAN, scale_factor=1.15), run_time=0.4)
        self.at("ccrp_mark")
        self.show(rel_mark, rt=0.4)
        self.at("ccrp_label")
        self.show(rr[2])
        self.at("ccrp_rel")
        F2 = self.flight(shift_path(P_WIND, XA1, AY + START_DY), "ccrp_rel", dur=1.3, clack=True)
        self.at("ccrp_rel", 1.3)
        track.glide(XA0, ("ccrp_rel", 1.4), ("ccrp_rel", 2.2))
        self.play(self.ping(tgt.get_center()), FadeOut(VGroup(*rr, rel_mark, fline)), FadeOut(F2.trail),
                  FadeOut(bomb), run_time=0.5, sfx=False)
        self.fl = None

        # ---------------------------------------------------------------- roquettes
        rk = [self.rowt("ROCKET", 0, 28), self.rowt("NO STEERING", 1, 24), self.rowt("IMPACT POINT", 2, 24)]
        pred_r = DashedVMobject(VMobject().set_points_as_corners(np.vstack([P_ROCKET[::2], P_ROCKET[-1:]])),
                                num_dashes=36, dashed_ratio=0.55).set_stroke(CYAN, 2.0).set_z_index(2)
        pip2 = self.pipper(follow=False).move_to([XT, GY + 0.5, 0])
        self.at("rocket")
        self.show(rk[0])
        self.at("rocket", 0.7)
        rocket.update(0)
        self.play(FadeIn(rocket), run_time=0.3, sfx=False)
        self.at("no_steer2")
        self.show(rk[1])
        self.at("pred_rocket")
        self.play(FadeIn(rk[2]), Create(pred_r), FadeIn(pip2), run_time=0.8)
        self.at("fire")
        F3 = self.flight(P_ROCKET, "fire", dur=1.1, clack=True, f0=700, f1=3800, thud=0.8)
        self.sfx("crackle", t=("fire", 1.1), dur=0.25, density=120, gain=0.25)
        self.at("fire", 1.1)
        track.glide(XA1, ("fire", 1.2), ("fire", 1.8))
        self.play(self.ping(tgt.get_center()), FadeOut(VGroup(*rk)), FadeOut(pred_r), FadeOut(pip2),
                  FadeOut(F3.trail), FadeOut(rocket), run_time=0.5, sfx=False)
        self.fl = None

        # ---------------------------------------------------------------- la prediction s'arrete au largage
        self.at("fire", 1.8)
        bomb.update(0)
        self.play(FadeIn(bomb), run_time=0.3, sfx=False)
        self.at("prediction")
        self.play(FadeIn(pred), run_time=0.4)
        r_rel, r_nc = self.rowt("RELEASE", 0, 28), self.rowt("NO CORRECTION", 1, 28)
        self.at("release_word")
        self.show(r_rel)
        self.at("go")
        G1 = self.flight(P_GUST, "go", dur=2.2, clack=True)
        self.play(pred.animate.set_stroke(opacity=0.45), run_time=0.3, sfx=False)
        pg = P_GUST[int(GUST[0] / 0.008)]
        gust_g = VGroup(*[Arrow([min(pg[0] + 1.5, 3.4), pg[1] + dy, 0], [min(pg[0] + 0.6, 2.5), pg[1] + dy, 0],
                                buff=0, color=CYAN, stroke_width=2.2, tip_length=0.14)
                          for dy in (-0.3, 0, 0.3)]).set_z_index(5)
        self.sfx("crackle", t=("go", 2.2 * GUST_FRAC - 0.25), dur=0.7, density=110, gain=0.5)
        self.at("go", 2.2 * GUST_FRAC - 0.2)
        self.play(FadeIn(gust_g), run_time=0.3, sfx=False)
        self.at("no_corr")
        self.play(FadeIn(r_nc), FadeOut(gust_g), run_time=0.4)
        self.at("go", 2.2)
        gap2 = ground_cote(P_GUST[-1][0], XT, GY)
        self.play(FadeIn(gap2), run_time=0.3, sfx=False)

        # ---------------------------------------------------------------- IMPACT = f(RELEASE)
        self.at("eq_clear")
        track.glide(XA0, ("eq_clear", 0.6), ("eq_clear", 0.601), smooth=False, sound=False)
        self.play(FadeOut(VGroup(r_rel, r_nc, gap2)), FadeOut(G1.trail), FadeOut(pred), FadeOut(tgt),
                  FadeOut(bomb), FadeOut(air), run_time=0.3, sfx=False)
        self.fl = None
        toks = [TX(s, 34, INK) for s in ["IMPACT", "=", "f(", "RELEASE", ")"]]
        toks[3].set_color(AMBER)
        eq = VGroup(*toks).arrange(RIGHT, buff=0.22).move_to([0, 1.4, 0]).set_z_index(9)
        if eq.width > config.frame_width - 0.8:
            eq.scale((config.frame_width - 0.8) / eq.width)
        self.sfx("rise", t="eq1", dur=max(self.cue("eq_box") - self.cue("eq1"), 0.5))
        for key, tk in zip(("eq1", "eq2", "eq3", "eq4", "eq5"), toks):
            self.at(key)
            self.show(tk, rt=0.25)
        box = SurroundingRectangle(eq, color=AMBER, buff=0.25, stroke_width=2.5)
        self.at("eq_box")
        self.sfx("impact")
        self.play(Create(box), run_time=0.4, sfx=False)
        nxt = TX("NEXT: GUIDANCE", 30, CYAN).move_to([0, -0.4, 0])
        self.at("nxt")
        self.show(nxt, rt=0.5)

        # ---------------------------------------------------------------- boucle : retour a l'image 0
        self.at("loop")
        self.sfx("whoosh", dur=0.8, f0=200, f1=900, gain=0.25)
        air.update(0)
        bomb.update(0)
        self.play(FadeOut(eq), FadeOut(box), FadeOut(nxt), FadeIn(air), FadeIn(bomb), FadeIn(hook_group),
                  run_time=0.8, sfx=False)
        self.at("end")
