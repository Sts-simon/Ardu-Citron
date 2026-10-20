"""
Serie "Types de guidage" - Episode 3 : le guidage infrarouge (la bombe voit la chaleur).

Meme DA blueprint et memes classes que l'episode 2 (Mon, Sight, Episode.put/drop/row), importees depuis ep02.py :
garde ep02.py et guidage.py dans le meme dossier.

    python3 guidage.py ep03.py script                        # texte a lire
    python3 guidage.py ep03.py gaps                          # ecarts minimaux entre cues
    python3 guidage.py ep03.py build --preview               # apercu SANS voix (cues estimees)
    python3 guidage.py ep03.py build --voice voix.wav        # rendu final synchronise sur ta voix

PORTEE : visualisation pedagogique et schematique. Aucun vol de projectile, aucune boucle de commande simulee,
aucun calcul de solution de tir : uniquement la physique (spectre, emission thermique, contraste, attenuation
atmospherique) et un schema en blocs.
---------------------------------------------------------------------------------------------
STORYBOARD (meme DA blueprint, 9:16, zone utile : haut libre 250 px, bas 450 px, droite 1 unite)

 Decor      Identique aux episodes 1 et 2 : sol hachure, batiment (cible), avion + bombe (a l'arret). Image 0 = derniere image.
 hook2      "THE BOMB CAN SEE" puis "HEAT" (cyan) sur "heat".
 ir/sensor  Spectre (axe en echelle log) : bloc VISIBLE blanc, bloc INFRARED cyan, "1800 . HERSCHEL" ; l'objectif IR
            apparait au nez (ligne de visee pointillee) et le moniteur s'allume.
 day/vis    Soleil et bloc VISIBLE attenues a 35 % : la bombe n'a plus besoin de lumiere visible.
 thermal    Fenetres atmospheriques 3-5 et 8-12 (cyan) ; "contrast" : tache chaude dans le moniteur + "DT".
 engine..   Trois icones (moteur, centrale, vehicule) avec ondes de chaleur cyan ; "dark" : decor attenue a 35 %,
 emit       seules les sources chaudes restent vives.
 trk..sig   Cadre de verrouillage (4 coins) sur la tache ; rangees TRACKER, THERMAL SIGNATURE ; ondes sur le batiment.
 guid..fol  Blocs SENSOR -> TRACKER -> GUIDANCE -> FINS (schema, pas de simulation).
 night      "NIGHT" / "!= PROTECTION" (decor attenue).
 mav..night2 Rangees AGM-65 MAVERICK IR, TOTAL DARKNESS, FIRE AND FORGET, EVEN AT NIGHT.
 limits..   Pluie, nuage, fumee entre le capteur et la cible : le bruit envahit le moniteur, la tache s'eteint,
 sig2       les coins se decrochent, "?".
 fn1..fn3   GUIDANCE = f( CONTRAST ) : CONTRAST en ambre + cadre ambre (l'unique touche chaude).
 Fin        "NEXT: GUIDANCE THAT DOESN'T RELY ON SIGHT" puis "GPS SATELLITE NAVIGATION" -> fondu vers l'image 0.
---------------------------------------------------------------------------------------------
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidage import *  # noqa: E402,F401,F403
import ep02 as E2  # noqa: E402
from ep02 import (Mon, Sight, P3, AY, GY, XA0, XT, BW, BH, MC, MW, MH, TGT)  # noqa: E402

# =============================================================================================
# VOIX OFF   (les mots sont ceux de la transcription : "1R" = "IR" prononce)
#   [[cle]]mot  /  [[cle@end]]mot  /  [[cle+0.2]]  /  {pause 1 s}
# =============================================================================================
SCRIPT = """
The bomb can see [[hook2]]heat. {pause 0.7 s}

Put an [[ir]]infrared [[sensor]]sensor in the nose, and the bomb no longer needs [[day]]daylight. {pause 0.6 s}

It doesn't look for [[vis]]visible light, it looks for [[thermal]]thermal [[contrast]]contrast. {pause 0.6 s}

An [[engine]]engine, a [[plant]]power plant, a [[vehicle]]vehicle in the [[dark]]dark, all [[emit]]emit heat. {pause 0.6 s}

The [[trk]]tracker locks onto the [[sig]]thermal signature, [[guid]]guidance moves the [[fins]]fins, and the bomb [[follows]]follows the target. {pause 0.7 s}

[[night]]Night no longer offers [[prot]]protection. {pause 0.6 s}

Weapons like the [[mav]]AGM-65 Maverick [[irm]]1R can strike in [[total]]total darkness. {pause 0.6 s}

[[fire]]Fire and forget, even at [[night2]]night. {pause 0.7 s}

But [[limits0]]thermal imagery still has [[limits]]limits. {pause 0.6 s}

Heavy [[rain]]rain, dense [[cloud]]clouds, or artificial [[smoke]]smoke can [[block]]block the heat [[sig2]]signature. {pause 0.5 s}

The [[fn1]]guidance is a [[fn2]]function of the [[fn3]]contrast. {pause 0.9 s}

[[nxt]]Next, {pause 0.6 s} [[guidt]]guidance that [[dnt]]doesn't rely on [[sight]]sight. {pause 0.6 s}

[[gps]]GPS and [[sat]]satellite [[loop@end+0.5]]navigation.
"""

RELATIVE = {
    "hook_out": "hook2+0.9",
    "end": "loop+1.0",
}

WHISPER_PROMPT = "AGM-65 Maverick, infrared, IR, thermal signature, thermal contrast, GPS, fire and forget."

# =============================================================================================
# GEOMETRIE DU SPECTRE (echelle logarithmique : 0,4 um -> x=-3,2 ; 12 um -> x=3,2)
# =============================================================================================
SY = 2.55


def lx(lam):
    return -1.475 + 4.333 * np.log10(lam)


# =============================================================================================
# OBJETS SPECIFIQUES
# =============================================================================================
class Glow:
    """Tache chaude du moniteur : ellipses cyan concentriques qui respirent ; s'eteint quand le bruit monte."""

    def __init__(self, sc, mon, z=10):
        self.sc, self.mon, self.op, self.cur = sc, mon, ValueTracker(0.0), 1.0
        self.spec = [(1.25, 0.80, 0.10), (0.90, 0.58, 0.18), (0.60, 0.38, 0.30), (0.30, 0.20, 0.55)]
        self.g = VGroup(*[Ellipse(width=w, height=h, stroke_width=0, fill_color=CYAN, fill_opacity=a)
                          for w, h, a in self.spec]).move_to(mon.c).set_z_index(z)
        self.g.add_updater(self._up)
        self._up(self.g, 0)
        sc.add(self.g)

    def _up(self, m, dt):
        nz = self.mon.nz.val()
        k = float(np.clip(1 - 1.25 * (nz - 0.15), 0, 1))
        v = self.op.get_value() * self.mon.vis.get_value() * k
        s = 1 + 0.05 * np.sin(self.sc.time * 5.0)
        m.scale(s / self.cur, about_point=self.mon.c)
        self.cur = s
        for e, (_, _, a) in zip(m, self.spec):
            e.set_fill(opacity=a * v)


class Waves:
    """Ondes de chaleur (infrarouge) : trois filets cyan qui montent en ondulant."""

    def __init__(self, sc, x, y0, h=0.8, n=3, w=0.42):
        self.sc, self.op = sc, ValueTracker(0.0)
        self.x, self.y0, self.h, self.n, self.w = x, y0, h, n, w
        self.g = VGroup().set_z_index(4)
        self.g.add_updater(self._up)
        sc.add(self.g)

    def _up(self, m, dt):
        o = self.op.get_value()
        if o < 0.02:
            m.become(VGroup()).set_z_index(4)
            return
        t = self.sc.time
        ys = np.linspace(0, self.h, 26)
        parts = VGroup()
        for k in range(self.n):
            xs = self.x + (k - (self.n - 1) / 2) * self.w
            pts = [P3(xs + 0.06 * np.sin(7 * y - 3.2 * t + 1.7 * k), self.y0 + y) for y in ys]
            parts.add(VMobject(stroke_color=CYAN, stroke_width=2.0, stroke_opacity=0.85 * o).set_points_as_corners(pts))
        m.become(parts).set_z_index(4)


class Rain:
    """Pluie schematique entre l'avion et le sol (traits blancs qui tombent)."""

    def __init__(self, sc, x0=-2.0, x1=1.8, y_top=-0.25, y_bot=GY + 0.1, n=26):
        self.sc, self.op = sc, ValueTracker(0.0)
        rng = np.random.default_rng(3)
        self.pts = [(rng.uniform(x0, x1), rng.uniform(y_bot, y_top), rng.uniform(0.8, 1.6)) for _ in range(n)]
        self.y_top, self.y_bot = y_top, y_bot
        self.g = VGroup().set_z_index(3)
        self.g.add_updater(self._up)
        sc.add(self.g)

    def _up(self, m, dt):
        o = self.op.get_value()
        if o < 0.02:
            m.become(VGroup()).set_z_index(3)
            return
        H, t, parts = self.y_top - self.y_bot, self.sc.time, VGroup()
        for x, y, sp in self.pts:
            py = self.y_top - ((self.y_top - y) + sp * t) % H
            parts.add(Line(P3(x, py), P3(x - 0.07, py - 0.24), color=INK, stroke_width=1.4, stroke_opacity=0.55 * o))
        m.become(parts).set_z_index(3)


def sun_icon():
    g = VGroup(Circle(radius=0.14, color=INK, stroke_width=2.0))
    for k in range(8):
        a = k * np.pi / 4
        g.add(Line(P3(0.22 * np.cos(a), 0.22 * np.sin(a)), P3(0.34 * np.cos(a), 0.34 * np.sin(a)),
                   color=INK, stroke_width=1.8))
    return g


def icon_engine():
    st = dict(stroke_color=INK, stroke_width=2.0, fill_color=BG, fill_opacity=1)
    body = Rectangle(width=0.7, height=0.4, **st)
    head = Rectangle(width=0.34, height=0.16, **st).next_to(body, UP, buff=0)
    pipe = Line(body.get_right(), body.get_right() + RIGHT * 0.22 + UP * 0.1, color=INK, stroke_width=2.0)
    return VGroup(body, head, pipe)


def icon_plant():
    st = dict(stroke_color=INK, stroke_width=2.0, fill_color=BG, fill_opacity=1)
    body = Rectangle(width=0.8, height=0.45, **st)
    chim = Rectangle(width=0.16, height=0.5, **st).next_to(body, UP, buff=0).align_to(body, RIGHT).shift(LEFT * 0.1)
    return VGroup(body, chim)


def icon_vehicle():
    st = dict(stroke_color=INK, stroke_width=2.0, fill_color=BG, fill_opacity=1)
    body = Rectangle(width=0.9, height=0.26, **st)
    cab = Rectangle(width=0.4, height=0.2, **st).next_to(body, UP, buff=0).shift(LEFT * 0.1)
    w1 = Circle(radius=0.1, **st).move_to(body.get_bottom() + LEFT * 0.28)
    w2 = Circle(radius=0.1, **st).move_to(body.get_bottom() + RIGHT * 0.28)
    return VGroup(body, cab, w1, w2)


def on_ground(vg, x):
    return vg.move_to([x, GY + vg.height / 2 + 0.02, 0]).set_z_index(4)


# =============================================================================================
# SCENE
# =============================================================================================
class Episode(E2.Episode):
    def construct(self):
        self.fl = None
        self.live = []
        track = self.track = Track(self, XA0)
        err1 = Track(self, 0.0)
        dim = ValueTracker(1.0)

        grid, border, sweep = self.background()
        ground = VGroup(
            Line([-4.25, GY, 0], [4.25, GY, 0], color=INK, stroke_width=2.2),
            *[Line([x, GY, 0], [x - 0.2, GY - 0.2, 0], color=INK, stroke_width=1.0, stroke_opacity=0.45)
              for x in np.arange(-4.0, 4.3, 0.35)]).set_z_index(1)
        building = VGroup(
            Rectangle(width=BW, height=BH, stroke_color=INK, stroke_width=2.4, fill_color=BG, fill_opacity=1),
            Line([-0.25, 0.08, 0], [-0.12, 0.08, 0]), Line([0.12, 0.08, 0], [0.25, 0.08, 0]),
            Line([0.0, -BH / 2, 0], [0.0, -0.05, 0])
        ).move_to([XT, GY + BH / 2, 0]).set_z_index(4)
        building[1:].set_stroke(INK, 1.4, opacity=0.75)

        def dim_ground(m, dt):
            d = dim.get_value()
            m[0].set_stroke(opacity=d)
            for h in m[1:]:
                h.set_stroke(opacity=0.45 * d)

        def dim_building(m, dt):
            d = dim.get_value()
            m[0].set_stroke(opacity=d)
            for h in m[1:]:
                h.set_stroke(opacity=0.75 * d)
        ground.add_updater(dim_ground)
        building.add_updater(dim_building)

        air = aircraft_shape().set_z_index(5).move_to([XA0, AY, 0])
        air.add_updater(lambda m, dt: m.move_to([track.val(), AY, 0]))
        bomb = self.make_proj(bomb_shape())
        hook_group, hook2 = make_hook("THE BOMB CAN SEE", "HEAT")

        mon = Mon(self, MC, MW, MH, err1, label=True)
        glow = Glow(self, mon)
        sight = Sight(self)
        rain = Rain(self)
        self.add(grid, border, sweep, ground, building, air, bomb, hook_group)

        # ---- spectre (echelle log) : VISIBLE blanc / INFRARED cyan / fenetres atmospheriques
        vis_blk = Rectangle(width=lx(0.7) - lx(0.4), height=0.34, stroke_color=INK, stroke_width=2.0
                            ).move_to([(lx(0.4) + lx(0.7)) / 2, SY, 0])
        ir_blk = Rectangle(width=lx(12) - lx(0.7), height=0.34, stroke_color=CYAN, stroke_width=2.0
                           ).move_to([(lx(0.7) + lx(12)) / 2, SY, 0])
        l_vis = TX("VISIBLE", 22, INK).move_to([vis_blk.get_center()[0], SY + 0.45, 0])
        l_ir = TX("INFRARED", 22, CYAN).move_to([ir_blk.get_center()[0], SY + 0.45, 0])
        l_unit = TX("WAVELENGTH (MICRONS)", 22, INK).move_to([-3.5, SY - 0.45, 0], aligned_edge=LEFT)
        spec = VGroup(vis_blk, ir_blk, l_vis, l_ir, l_unit).set_z_index(9)
        vis_grp = VGroup(vis_blk, l_vis)
        sun = sun_icon().move_to([vis_blk.get_center()[0], 3.55, 0]).set_z_index(9)
        sun.set_opacity(0.35)
        herschel = TX("1800 · HERSCHEL", 22, CYAN).move_to([-3.5, 1.65, 0], aligned_edge=LEFT).set_z_index(9)
        wins = VGroup(
            Rectangle(width=lx(5) - lx(3), height=0.34, stroke_width=0, fill_color=CYAN, fill_opacity=0.35
                      ).move_to([(lx(3) + lx(5)) / 2, SY, 0]),
            Rectangle(width=lx(12) - lx(8), height=0.34, stroke_width=0, fill_color=CYAN, fill_opacity=0.35
                      ).move_to([(lx(8) + lx(12)) / 2, SY, 0]),
            TX("3–5", 22, CYAN).move_to([(lx(3) + lx(5)) / 2, SY - 0.45, 0]),
            TX("8–12", 22, CYAN).move_to([(lx(8) + lx(12)) / 2 - 0.05, SY - 0.45, 0])).set_z_index(9)
        dtl = TX("ΔT", 34, CYAN).move_to([MC[0] + 2.2, MC[1], 0]).set_z_index(9)

        # ---- icones de sources chaudes + ondes
        ic = [on_ground(icon_engine(), -2.3), on_ground(icon_plant(), -0.5), on_ground(icon_vehicle(), 1.15)]
        wv = [Waves(self, ic[0].get_center()[0], ic[0].get_top()[1] + 0.05),
              Waves(self, ic[1].get_center()[0], ic[1].get_top()[1] + 0.05),
              Waves(self, ic[2].get_center()[0], ic[2].get_top()[1] + 0.05)]
        wv_b = Waves(self, XT, GY + BH + 0.05, h=0.9)

        # ---------------------------------------------------------------- accroche
        self.at("hook2")
        self.play(FadeIn(hook2), run_time=0.4)
        self.at("hook_out")
        self.play(FadeOut(hook_group), FadeOut(hook2), run_time=0.3)

        # ---------------------------------------------------------------- spectre : l'infrarouge
        self.at("ir")
        self.put(spec, rt=0.45)
        self.at("sensor")
        self.sfx("lock", t=("sensor", 0.2))
        self.play(self.ping(S0n()), mon.vis.animate.set_value(1), sight.op.animate.set_value(0.6),
                  run_time=0.45, sfx=False)
        self.at("sensor", 0.9)
        self.put(herschel, rt=0.4)

        # ---------------------------------------------------------------- plus besoin de lumiere visible
        self.at("day")
        self.put(sun, anims=[vis_blk.animate.set_stroke(opacity=0.35), l_vis.animate.set_opacity(0.35)], rt=0.5)
        self.at("vis")
        self.play(self.ping(vis_blk.get_center()), run_time=0.5)

        # ---------------------------------------------------------------- contraste thermique
        self.at("thermal")
        self.put(wins, anims=[Indicate(l_ir, color=CYAN, scale_factor=1.15)], rt=0.38)
        self.at("contrast")
        mon.nz.glide(0.15, "contrast", ("contrast", 0.6), sound=False)
        self.sfx("tick")
        self.live.append(dtl)
        self.play(glow.op.animate.set_value(0.75), FadeIn(dtl), run_time=0.4, sfx=False)

        # ---------------------------------------------------------------- toutes les sources emettent de la chaleur
        self.at("engine", -0.5)
        self.drop(rt=0.4)
        self.at("engine")
        self.put(ic[0], anims=[wv[0].op.animate.set_value(0.6)], rt=0.4)
        self.at("plant")
        self.put(ic[1], anims=[wv[1].op.animate.set_value(0.6)], rt=0.4)
        self.at("vehicle")
        self.put(ic[2], anims=[wv[2].op.animate.set_value(0.6)], rt=0.4)
        self.at("dark")
        self.sfx("tick")
        self.play(dim.animate.set_value(0.35), run_time=0.5, sfx=False)
        self.at("emit")
        self.sfx("whoosh", t="emit", dur=0.8, f0=120, f1=420, gain=0.2)
        self.play(*[w.op.animate.set_value(1.0) for w in wv], glow.op.animate.set_value(1.0), run_time=0.5,
                  sfx=False)

        # ---------------------------------------------------------------- le tracker accroche la signature thermique
        r_trk, r_sig = self.row("TRACKER", 0, 28), self.row("THERMAL SIGNATURE", 1, 26)
        self.at("trk", -0.6)
        self.drop(*[w.op.animate.set_value(0) for w in wv], extra=ic, rt=0.4)
        self.at("trk")
        self.sfx("lock", t=("trk", 0.25))
        self.put(r_trk, anims=[mon.lk.animate.set_value(1), mon.lkp.animate.set_value(0)], rt=0.4, sfx=False)
        self.at("sig")
        self.put(r_sig, anims=[wv_b.op.animate.set_value(0.9), self.ping(MC)], rt=0.4)

        # ---------------------------------------------------------------- schema en blocs (pas de simulation)
        ys = [2.9, 2.22, 1.54, 0.86]
        blocks = [block(t, 1.8, 0.48, 22).move_to([-0.9, y, 0]).set_z_index(9)
                  for t, y in zip(["SENSOR", "TRACKER", "GUIDANCE", "FINS"], ys)]
        arrows = [arr([-0.9, ys[i] - 0.24, 0], [-0.9, ys[i + 1] + 0.24, 0]).set_z_index(9) for i in range(3)]
        self.at("guid")
        self.put(blocks[0], arrows[0], blocks[1], arrows[1], blocks[2], clear=True, rt=0.4)
        self.at("guid", 0.45)
        self.play(Indicate(blocks[2], color=CYAN, scale_factor=1.12), run_time=0.35)
        self.at("fins")
        self.put(arrows[2], blocks[3], rt=0.35)
        self.at("fins", 0.4)
        self.play(Indicate(blocks[3], color=CYAN, scale_factor=1.12), run_time=0.4)
        self.at("follows")
        self.play(self.ping(TGT), run_time=0.5)

        # ---------------------------------------------------------------- la nuit n'est plus une protection
        r_n, r_p = self.row("NIGHT", 0, 28, CYAN), self.row("≠ PROTECTION", 1, 26)
        self.at("night", -0.7)
        self.drop(wv_b.op.animate.set_value(0), rt=0.5)
        self.at("night")
        self.put(r_n, rt=0.4)
        self.at("prot")
        self.put(r_p, rt=0.4)

        # ---------------------------------------------------------------- Maverick IR : frappe de nuit
        r_m = self.row("AGM-65 MAVERICK", 0, 26)
        r_ir = TX("IR", 26, CYAN).next_to(r_m, RIGHT, buff=0.24).set_z_index(9)
        r_t, r_ff, r_en = self.row("TOTAL DARKNESS", 1, 26), self.row("FIRE AND FORGET", 2, 26), \
            self.row("EVEN AT NIGHT", 3, 26)
        self.at("mav")
        self.put(r_m, clear=True, rt=0.4)
        self.at("irm")
        self.live.append(r_ir)
        self.play(FadeIn(r_ir), run_time=0.3)
        self.play(Indicate(r_ir, color=CYAN, scale_factor=1.4), run_time=0.4)
        self.at("total")
        self.put(r_t, rt=0.4)
        self.at("fire")
        self.put(r_ff, rt=0.4)
        self.at("night2")
        self.put(r_en, rt=0.4)

        # ---------------------------------------------------------------- limites : pluie, nuages, fumee
        self.at("limits0")
        self.drop(dim.animate.set_value(1.0), rt=0.5)
        r_l, r_r = self.row("LIMITS", 0, 28, CYAN), self.row("RAIN", 1, 26)
        r_c, r_s = self.row("CLOUDS", 2, 26), self.row("SMOKE", 3, 26)
        cloud = Union(Circle(radius=0.30).move_to([-0.3, -1.25, 0]), Circle(radius=0.40).move_to([0.1, -1.1, 0]),
                      Circle(radius=0.32).move_to([0.5, -1.22, 0]), Rectangle(width=1.3, height=0.3).move_to([0.1, -1.42, 0]),
                      stroke_color=INK, stroke_width=2.0, fill_color=BG, fill_opacity=1).set_z_index(3.5)
        puffs = VGroup(*[DashedVMobject(Circle(radius=r, color=INK, stroke_width=1.8, stroke_opacity=0.7),
                                        num_dashes=max(10, int(r * 40))).move_to(p)
                         for r, p in [(0.16, [1.0, -3.35, 0]), (0.24, [0.95, -3.0, 0]), (0.33, [0.85, -2.6, 0]),
                                      (0.42, [0.75, -2.15, 0]), (0.52, [0.65, -1.65, 0])]]).set_z_index(3.5)
        self.at("limits")
        self.put(r_l, rt=0.4)
        self.at("rain")
        self.sfx("crackle", t="rain", dur=1.0, density=70, gain=0.25)
        self.put(r_r, anims=[rain.op.animate.set_value(1)], rt=0.5, sfx=True)
        self.at("cloud")
        self.put(r_c, cloud, rt=0.5)
        self.at("smoke")
        self.sfx("whoosh", t="smoke", dur=0.7, f0=150, f1=300, gain=0.2)
        self.put(r_s, puffs, rt=0.5)
        self.at("block")
        mon.nz.glide(1.0, "block", ("sig2", 0.4), sound=False)
        mon.det.glide(1.0, "block", ("sig2", 0.5), sound=False)
        self.sfx("crackle", t="block", dur=1.1, density=140, gain=0.4)
        self.at("sig2")
        self.sfx("clack", t="sig2", gain=0.5)
        self.play(mon.q.animate.set_value(1), run_time=0.35, sfx="tick")

        # ---------------------------------------------------------------- GUIDANCE = f( CONTRAST )
        self.at("fn1", -0.6)
        self.drop(FadeOut(air), FadeOut(bomb), mon.vis.animate.set_value(0), sight.op.animate.set_value(0),
                  rain.op.animate.set_value(0), glow.op.animate.set_value(0), rt=0.4)
        toks = [TX(s, 34, INK) for s in ["GUIDANCE", "=", "f(", "CONTRAST", ")"]]
        toks[3].set_color(AMBER)
        eq = VGroup(*toks).arrange(RIGHT, buff=0.22).move_to([0, 1.4, 0]).set_z_index(9)
        if eq.width > config.frame_width - 0.8:
            eq.scale((config.frame_width - 0.8) / eq.width)
        self.sfx("rise", t="fn1", dur=max(self.cue("fn3") + 0.6 - self.cue("fn1"), 0.5))
        self.at("fn1")
        self.show(toks[0], rt=0.25)
        self.show(toks[1], rt=0.2)
        self.at("fn2")
        self.show(toks[2], rt=0.25)
        self.at("fn3")
        self.show(toks[3], rt=0.25)
        self.show(toks[4], rt=0.15)
        box = SurroundingRectangle(eq, color=AMBER, buff=0.25, stroke_width=2.5)
        self.at("fn3", 0.6)
        self.sfx("impact")
        self.play(Create(box), run_time=0.4, sfx=False)

        # ---------------------------------------------------------------- suite : sans la vue
        t_next, t_gt = TX("NEXT:", 28, CYAN), TX("GUIDANCE THAT", 28, CYAN)
        t_dnt, t_sight = TX("DOESN'T RELY ON", 28, CYAN), TX("SIGHT", 28, CYAN)
        t_gps, t_sat = TX("GPS", 28, CYAN), TX("SATELLITE NAVIGATION", 28, CYAN)
        l1 = VGroup(t_next, t_gt).arrange(RIGHT, buff=0.24).move_to([0, -0.3, 0])
        l2 = VGroup(t_dnt, t_sight).arrange(RIGHT, buff=0.24).move_to([0, -0.85, 0])
        l3 = VGroup(t_gps, t_sat).arrange(RIGHT, buff=0.24).move_to([0, -1.7, 0])
        VGroup(l1, l2, l3).set_z_index(9)
        self.at("nxt")
        self.show(t_next, rt=0.4)
        self.at("guidt")
        self.show(t_gt, rt=0.4)
        self.at("dnt")
        self.show(t_dnt, rt=0.4)
        self.at("sight")
        self.play(FadeIn(t_sight), run_time=0.3)
        self.play(Indicate(t_sight, color=CYAN, scale_factor=1.25), run_time=0.4)
        self.at("gps")
        self.show(t_gps, rt=0.35)
        self.at("sat")
        self.show(t_sat, rt=0.4)

        # ---------------------------------------------------------------- boucle : retour a l'image 0
        self.at("loop")
        self.sfx("whoosh", dur=0.8, f0=200, f1=900, gain=0.25)
        air.update(0)
        bomb.update(0)
        self.play(FadeOut(eq), FadeOut(box), *[FadeOut(t) for t in (t_next, t_gt, t_dnt, t_sight, t_gps, t_sat)],
                  FadeIn(air), FadeIn(bomb), FadeIn(hook_group), run_time=0.8, sfx=False)
        self.at("end")


def S0n():
    """Nez de la bombe a l'arret (point ou apparait l'objectif IR)."""
    return P3(XA0 + 0.32, AY - 0.30)
