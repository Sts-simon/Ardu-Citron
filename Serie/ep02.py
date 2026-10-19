"""
Serie "Types de guidage" - Episode 2 : le guidage TV (bombes et missiles a camera).

Etat : SCRIPT + SCENE + BRUITAGES ecrits (les bruitages sont deduits de la scene par guidage.py).

    python3 guidage.py ep02.py script                        # texte a lire
    python3 guidage.py ep02.py gaps                          # ecarts minimaux entre cues
    python3 guidage.py ep02.py build --preview               # apercu SANS voix (cues estimees)
    python3 guidage.py ep02.py build --voice voix.wav        # rendu final synchronise sur ta voix

---------------------------------------------------------------------------------------------
STORYBOARD (meme DA blueprint, format 9:16, zone utile : haut libre 250 px, bas 450 px, droite 1 unite)

 Decor      Vue de cote comme l'episode 1 : sol hachure en bas, avion en haut a gauche, bombe sous l'avion,
            cible = petit batiment (rectangle blanc) au sol. Image 0 = derniere image.
 Moniteur   Ecran = rectangle arrondi (trait blanc 1,8 px) en haut de l'image (y ~ 3,6 a 5,4), centre : reticule
            blanc fin, cible = rectangle, "cadre de verrouillage" = quatre coins cyan.

 Accroche   "THE BOMB CAN SEE" puis "THE TARGET" (cyan) sur le mot "target".
 camera     Cone de vision en pointilles cyan qui part du nez de la bombe vers la cible. Rangee "TV CAMERA".
 picture    Le moniteur apparait : la cible dans l'image, decalee du centre.
 tracker    Les quatre coins cyan se calent sur la cible (verrouillage).
 far/center Cote cyan entre le centre du reticule et la cible, avec delta (comme l'episode 1) .
 error      Delta s'allume ; rangee "ERROR".
 guid..back Bloc CAMERA -> TRACKER -> GUIDANCE -> FINS (blocs arrondis, fleches fines) ; la bombe se met a
            voler vers la cible, la cible dans le moniteur glisse vers le centre : delta -> 0.
 zero       delta = 0 : la cible est au centre, la bombe arrive droit sur la cible.
 walleye    Avion + bombe ; verrouillage au sol sur l'ecran ("CONTRAST") puis largage : "NO HUMAN",
            "FIRE AND FORGET".
 gbu        Liaison en pointilles bombe -> avion, petit moniteur dans l'avion, "DATA LINK", "OPERATOR",
            "MAN IN THE LOOP" (boucle qui passe par l'avion, comme la boucle de l'episode 0).
 light..lost Le moniteur se remplit de points cyan qui vibrent (incertitude), les coins se decrochent, un "?"
            apparait : la cible est perdue. Rangees "LIGHT", "NIGHT", "SMOKE", "FOG".
 fn1..fn3   GUIDANCE = f( IMAGE ) : IMAGE en ambre + cadre ambre (l'unique touche chaude, comme
            IMPACT = f(RELEASE) a l'episode 1).
 Fin        "NEXT: GUIDANCE THAT SEES HEAT" -> fondu vers l'image 0 (boucle).
---------------------------------------------------------------------------------------------
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guidage import *  # noqa: E402,F401,F403

# =============================================================================================
# VOIX OFF
#   [[cle]]mot       l'image "cle" se declenche au debut du mot qui suit
#   [[cle@end]]mot   ... a la fin du mot      [[cle+0.2]] / [[cle@end-0.1]] : decalage en secondes
#   {pause 1 s}      silence a laisser a la lecture ; ignore par l'analyse
# =============================================================================================
SCRIPT = """
The bomb can see the [[hook2]]target.

Put a [[camera]]TV camera in the nose, and the bomb is no longer [[blind]]blind. {pause 0.5 s}

The camera sends a [[picture]]picture. A [[tracker]]tracker finds the target in it, and measures how [[far]]far it is from the [[center]]center.

That [[dist]]distance is the [[error]]error. [[guid]]Guidance turns it into commands, the [[fins]]fins move, and the target comes [[back]]back to the center. {pause 0.8 s}

Keep the error at [[zero]]zero, and the bomb flies [[straight]]straight at what it sees.

Weapons like the [[walleye]]Walleye lock onto the [[contrast]]contrast of the target before launch. After release, [[nohuman]]no human is needed: fire and [[fnf@end]]forget. {pause 0.8 s}

Weapons like the [[gbu]]GBU-15 send the picture back by [[datalink]]data link. An [[operator]]operator watches and steers: a [[mil]]man in the loop. {pause 0.8 s}

But a camera needs [[light]]light and contrast. [[night]]Night, [[smoke]]smoke, [[fog]]fog: the picture is [[lost]]lost, and so is the target. {pause 0.8 s}

The [[fn1]]guidance is a [[fn2]]function of the [[fn3]]image. {pause 0.6 s}

[[nxt]]Next: guidance that sees [[loop@end+0.5]]heat.
"""

# Cues sans mot dedie (a completer quand la scene sera ecrite)
RELATIVE = {
    "hook_out": "hook2+0.9",
    "end": "loop+1.0",
}

# Aide a la transcription (mots rares)
WHISPER_PROMPT = "Walleye, GBU-15, TV camera, tracker, data link, fire and forget, man in the loop."


# =============================================================================================
# DECOR ET TRAJECTOIRES (illustratifs)
# =============================================================================================
AY, GY = 0.35, -3.7                  # avion / sol (l'avion est plus bas qu'a l'episode 1 : le moniteur est en haut)
XA0, START_DY = -3.0, -0.30
XT = 2.7                             # abscisse du batiment-cible
BW, BH = 0.9, 0.55
MC, MW, MH = np.array([0.0, 4.55, 0.0]), 3.4, 1.7      # grand moniteur (haut, centre)
ROWS_Y0, ROWS_DY = 2.9, 0.44         # colonne de mots-cles sous le moniteur


def P3(x, y):
    return np.array([x, y, 0.0])


S0 = P3(XA0, AY + START_DY)
TGT = P3(XT, GY + 0.3)


def hdg(deg):
    a = np.radians(deg)
    return P3(np.cos(a), np.sin(a))


def bez(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def quad(p0, p1, p2, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


# guidage TV : virage (43 % du temps) puis ligne droite sur la cible
_A = P3(-0.2, -0.6)
_dA = (TGT - _A) / np.linalg.norm(TGT - _A)
_NG = 600
_NC = int(_NG * 0.43)
P_GUIDED = np.vstack([bez(S0, S0 + 1.4 * hdg(-10), _A - 1.0 * _dA, _A, _NC),
                      np.linspace(_A, TGT, _NG - _NC + 1)[1:]])
# Walleye : verrouille avant largage, descente directe
P_WALLEYE = bez(S0, S0 + 1.8 * hdg(-12), TGT + P3(-1.1, 1.6), TGT, 400)
# GBU-15 : plane, l'operateur corrige (leger S)
P_GBU = bez(S0, P3(-1.5, 0.15), P3(1.7, -1.15), TGT, 500)
X_MISS = 1.15


def lost_path(frac, n=500):
    """Guidage normal jusqu'a la perte de cible, puis la bombe decroche et tombe court."""
    L = P3(0.1, -0.9)
    n1 = max(int(n * frac), 60)
    p1 = bez(S0, S0 + 1.4 * hdg(-10), L - 1.0 * hdg(-40), L, n1)
    p2 = quad(L, L + 0.6 * hdg(-40), P3(X_MISS, GY + 0.06), n - n1 + 1)[1:]
    return np.vstack([p1, p2])


# =============================================================================================
# OBJETS SPECIFIQUES A L'EPISODE
# =============================================================================================
CORN = [(1, 1), (-1, 1), (-1, -1), (1, -1)]


class Mon:
    """
    Moniteur de la camera : cadre arrondi, reticule fin, cible (rectangle) decalee de err*off0,
    cadre de verrouillage (4 coins cyan), cote + delta, nuage de points (incertitude), '?'.
    Tout est pilote par des trackers (opacites) et par `err` (Track) : un seul updater met tout a jour.
    """

    def __init__(self, sc, c, w, h, err, label=True, z=8):
        self.sc, self.c, self.err, self.label = sc, np.array(c, float), err, label
        s = self.s = w / 3.4
        self.hw, self.hh = w / 2, h / 2
        self.off0 = np.array([0.235 * w, -0.294 * h, 0.0])
        self.tw, self.th, self.arm = 0.5 * s, 0.3 * s, 0.16 * s
        self.nz, self.det = Track(sc, 0.0), Track(sc, 0.0)
        self.vis, self.tg, self.lk, self.ct, self.lb, self.q = [ValueTracker(0.0) for _ in range(6)]
        self.lkp = ValueTracker(0.45)                     # marge supplementaire du cadre (0 = verrouille)
        self.rng = np.random.default_rng(7)

        self.frame = RoundedRectangle(corner_radius=0.1 * s + 0.04, width=w, height=h, stroke_color=INK,
                                      stroke_width=1.8, fill_color=BG, fill_opacity=1).move_to(self.c).set_z_index(z)
        self.ret = VGroup(*[Line(self.c + d * 0.10 * s, self.c + d * 0.38 * s, color=INK, stroke_width=1.4)
                            for d in (UP, DOWN, LEFT, RIGHT)]).set_z_index(z + 1)
        self.cote = VGroup().set_z_index(z + 1)
        self.rect = Rectangle(width=self.tw, height=self.th, stroke_color=INK, stroke_width=2.0,
                              fill_color=BG, fill_opacity=1).set_z_index(z + 2)
        self.base = [P3(self.rng.uniform(-self.hw + 0.1, self.hw - 0.1),
                        self.rng.uniform(-self.hh + 0.1, self.hh - 0.1)) for _ in range(80)]
        self.dots = VGroup(*[Dot(self.c + b, radius=0.03 * max(s, 0.55), color=CYAN) for b in self.base]
                           ).set_z_index(z + 3)
        self.corners = VGroup(*[VMobject(stroke_color=CYAN, stroke_width=2.6).set_points_as_corners(
            [P3(-sx * self.arm, 0), P3(0, 0), P3(0, -sy * self.arm)]) for sx, sy in CORN]).set_z_index(z + 4)
        self.lab = TX("Δ", 30, CYAN).set_z_index(z + 4)
        self.qm = TX("?", int(46 * max(s, 0.6)), CYAN).move_to(self.c).set_z_index(z + 5)
        self.master = Mobject()
        self.master.add_updater(self._up)
        self._up(self.master, 0)
        sc.add(self.frame, self.ret, self.cote, self.rect, self.dots, self.corners, self.lab, self.qm,
               self.master)

    def parts(self):
        return [self.frame, self.ret, self.rect, self.dots, self.corners, self.lab, self.qm]

    def _up(self, m, dt):
        v, e, nz, det = self.vis.get_value(), self.err.val(), self.nz.val(), self.det.val()
        s, c = self.s, self.c
        tc = c + self.off0 * e
        self.frame.set_stroke(opacity=v).set_fill(opacity=v)
        self.ret.set_stroke(opacity=v)
        tg = self.tg.get_value() * v * max(0.0, 1 - 1.1 * nz)
        self.rect.move_to(tc).set_stroke(opacity=tg).set_fill(opacity=tg)
        # nuage de points qui vibre
        for i, (d, b) in enumerate(zip(self.dots, self.base)):
            d.move_to(c + b + P3(*self.rng.normal(0, 0.035 * s + 0.01, 2)))
            d.set_opacity(float(np.clip(nz * len(self.base) - i, 0, 1)) * 0.85 * v)
        # cadre de verrouillage (se decroche avec det)
        pad = 0.12 * s + self.lkp.get_value() * s
        lk = self.lk.get_value() * v * (1 - 0.55 * det)
        for (sx, sy), cn in zip(CORN, self.corners):
            vert = tc + P3(sx * (self.tw / 2 + pad), sy * (self.th / 2 + pad))
            vert = vert + det * s * 0.55 * P3(sx + 0.35 * sy, sy - 0.35 * sx)
            vert = vert + det * P3(*self.rng.normal(0, 0.02 * s, 2))
            cn.move_to(vert + P3(-sx * self.arm / 2, -sy * self.arm / 2)).set_stroke(opacity=lk)
        # cote entre le centre du reticule et la cible
        d = tc - c
        ln = float(np.linalg.norm(d))
        op = self.ct.get_value() * v * min(1.0, ln / 0.5)
        if ln > 0.25 * s and op > 0.02:
            u = d / ln
            te = min(self.tw / 2 / max(abs(u[0]), 1e-6), self.th / 2 / max(abs(u[1]), 1e-6))
            p, q = c + u * 0.14 * s, tc - u * (te + 0.05 * s)
            ar = DoubleArrow(p, q, buff=0, stroke_width=2.2, tip_length=0.12 * max(s, 0.5), color=CYAN,
                             max_tip_length_to_length_ratio=0.45)
            self.cote.become(VGroup(ar).set_opacity(op)).set_z_index(self.frame.z_index + 1)
            if self.label:
                n = P3(-u[1], u[0])
                n = n if n[1] > 0 else -n
                self.lab.move_to((p + q) / 2 + n * 0.3).set_opacity(self.lb.get_value() * op)
            else:
                self.lab.set_opacity(0)
        else:
            self.cote.become(VGroup()).set_z_index(self.frame.z_index + 1)
            self.lab.set_opacity(0)
        self.qm.set_opacity(self.q.get_value() * v)


class Sight:
    """Ligne de visee de la camera : deux pointilles du nez de la bombe vers les coins du batiment + objectif."""

    def __init__(self, sc):
        self.sc, self.op = sc, ValueTracker(0.0)
        self.g = VGroup().set_z_index(3)
        self.g.add_updater(self._up)
        sc.add(self.g)

    def _up(self, m, dt):
        o = self.op.get_value()
        if o < 0.02:
            m.become(VGroup()).set_z_index(3)
            return
        p, ang = self.sc.bomb_pose()
        nose = p + 0.32 * hdg(np.degrees(ang))
        k = dict(dash_length=0.11, stroke_width=1.6, color=CYAN)
        parts = VGroup(DashedLine(nose, P3(XT - BW / 2, GY + BH), **k),
                       DashedLine(nose, P3(XT + BW / 2, GY + BH), **k),
                       Circle(radius=0.07, color=CYAN, stroke_width=2.2).move_to(nose))
        m.become(parts.set_stroke(opacity=o * 0.8)).set_z_index(3)


class Link:
    """Liaison de donnees bombe <-> avion : image vers l'avion (cyan), ordres vers la bombe (blanc)."""

    def __init__(self, sc):
        self.sc, self.a, self.b = sc, ValueTracker(0.0), ValueTracker(0.0)
        self.g = VGroup().set_z_index(4)
        self.g.add_updater(self._up)
        sc.add(self.g)

    def _up(self, m, dt):
        a, b = self.a.get_value(), self.b.get_value()
        if a < 0.02 and b < 0.02:
            m.become(VGroup()).set_z_index(4)
            return
        p, _ = self.sc.bomb_pose()
        q = P3(self.sc.track.val(), AY)
        L = float(np.linalg.norm(p - q))
        if L < 0.2:
            m.become(VGroup()).set_z_index(4)
            return
        u = (p - q) / L
        n = P3(-u[1], u[0])
        ph = self.sc.time * 0.45
        parts = VGroup()
        if a > 0.02:
            pa, qa = p + 0.1 * n, q + 0.1 * n
            parts.add(DashedLine(pa, qa, dash_length=0.11, stroke_width=1.8, color=CYAN).set_stroke(opacity=a))
            for k in range(4):
                parts.add(Dot(pa + (qa - pa) * ((ph + k / 4) % 1), radius=0.05, color=CYAN).set_opacity(a))
        if b > 0.02:
            pb, qb = q - 0.1 * n, p - 0.1 * n
            parts.add(DashedLine(pb, qb, dash_length=0.11, stroke_width=1.6, color=INK).set_stroke(opacity=0.6 * b))
            for k in range(4):
                parts.add(Dot(pb + (qb - pb) * ((ph + k / 4) % 1), radius=0.045, color=INK).set_opacity(b))
        m.become(parts).set_z_index(4)


# =============================================================================================
# SCENE
# =============================================================================================
class Episode(BlueprintScene):
    # ---- projectile : sous l'avion, ou sur la trajectoire en cours
    def bomb_pose(self):
        if self.fl is None:
            return P3(self.track.val(), AY + START_DY), 0.0
        return self.fl.pos(), self.fl.angle()

    def make_proj(self, shape):
        shape.set_z_index(6)
        shape.ang = 0.0

        def up(m, dt):
            p, ang = self.bomb_pose()
            m.rotate(ang - m.ang, about_point=m.get_center())
            m.ang = ang
            m.move_to(p)
        shape.add_updater(up)
        up(shape, 0)
        return shape

    # ---- aides de mise en scene
    def row(self, s, i, size=26, color=INK):
        return TX(s, size, color).move_to([EQ_X0, ROWS_Y0 - ROWS_DY * i, 0], aligned_edge=LEFT).set_z_index(9)

    def put(self, *mobs, anims=(), rt=0.4, clear=False, sfx=True):
        outs = [FadeOut(m) for m in self.live] if clear else []
        self.live = list(mobs) if clear else self.live + list(mobs)
        self.play(*[FadeIn(m) for m in mobs], *outs, *anims, run_time=rt, sfx=sfx)

    def drop(self, *anims, rt=0.4, extra=()):
        outs = [FadeOut(m) for m in self.live] + [FadeOut(m) for m in extra]
        self.live = []
        if outs or anims:
            self.play(*outs, *anims, run_time=rt, sfx=False)

    def lost_frac(self, t0, t1):
        dur = t1 - t0
        if dur <= 1e-6:
            return 0.7
        return float(np.clip((self.cue("lost") - t0) / dur, 0.3, 0.9))

    # ---- timeline
    def construct(self):
        self.fl = None
        self.live = []
        track = self.track = Track(self, XA0)
        err1, err2 = Track(self, 1.0), Track(self, 1.0)

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
        air = aircraft_shape().set_z_index(5).move_to([XA0, AY, 0])
        air.add_updater(lambda m, dt: m.move_to([track.val(), AY, 0]))
        bomb = self.make_proj(bomb_shape())
        hook_group, hook2 = make_hook("THE BOMB CAN SEE", "THE TARGET")

        mon = Mon(self, MC, MW, MH, err1, label=True)
        mini = Mon(self, [-1.1, 0.7, 0], 1.3, 0.78, err2, label=False)
        sight, link = Sight(self), Link(self)
        self.add(grid, border, sweep, ground, building, air, bomb, hook_group)

        # ---------------------------------------------------------------- accroche
        self.at("hook2")
        self.play(FadeIn(hook2), run_time=0.4)
        self.at("hook_out")
        self.play(FadeOut(hook_group), FadeOut(hook2), run_time=0.3)

        # ---------------------------------------------------------------- camera : la bombe n'est plus aveugle
        r_cam, r_trk, r_err = self.row("TV CAMERA", 0, 28), self.row("TRACKER", 1, 26), self.row("ERROR", 2, 28)
        self.at("camera")
        self.put(r_cam, anims=[sight.op.animate.set_value(0.6)], rt=0.5)
        self.at("blind")
        self.play(self.ping(S0 + P3(0.32, 0)), sight.op.animate.set_value(1.0), run_time=0.5)

        # ---------------------------------------------------------------- picture / tracker
        self.at("picture")
        self.sfx("tick")
        self.play(mon.vis.animate.set_value(1), mon.tg.animate.set_value(1), run_time=0.4, sfx=False)
        self.at("tracker")
        self.sfx("lock", t=("tracker", 0.25))
        self.put(r_trk, anims=[mon.lk.animate.set_value(1), mon.lkp.animate.set_value(0)], rt=0.4, sfx=False)

        # ---------------------------------------------------------------- distance au centre = erreur
        self.at("far")
        self.sfx("tick")
        self.play(mon.ct.animate.set_value(1), run_time=0.35, sfx=False)
        self.at("center")
        self.play(self.ping(MC), run_time=0.5)
        self.at("dist")
        self.sfx("tick")
        self.play(mon.lb.animate.set_value(1), run_time=0.35, sfx=False)
        self.at("error")
        self.put(r_err, anims=[Indicate(mon.lab, color=CYAN, scale_factor=1.5)], rt=0.5)

        # ---------------------------------------------------------------- guidage -> gouvernes -> l'erreur s'annule
        ys = [2.9, 2.22, 1.54, 0.86]
        blocks = [block(t, 1.8, 0.48, 22).move_to([-0.9, y, 0]).set_z_index(9)
                  for t, y in zip(["CAMERA", "TRACKER", "GUIDANCE", "FINS"], ys)]
        arrows = [arr([-0.9, ys[i] - 0.24, 0], [-0.9, ys[i + 1] + 0.24, 0]).set_z_index(9) for i in range(3)]
        self.at("guid")
        F1 = self.flight(P_GUIDED, "guid", end=("straight", 0.5), clack=True, gain=0.35)
        err1.glide(0.0, "guid", "back", sound=False)
        self.put(blocks[0], arrows[0], blocks[1], arrows[1], blocks[2], clear=True, rt=0.45)
        self.at("guid", 0.7)
        self.play(Indicate(blocks[2], color=CYAN, scale_factor=1.12), run_time=0.5)
        self.at("fins")
        self.put(arrows[2], blocks[3], rt=0.4)
        self.at("fins", 0.5)
        self.play(Indicate(blocks[3], color=CYAN, scale_factor=1.12), run_time=0.5)
        self.at("back")
        self.drop(rt=0.5)

        # ---------------------------------------------------------------- delta = 0
        r_zero = TX("Δ = 0 ✓", 34, CYAN).move_to([EQ_X0, ROWS_Y0, 0], aligned_edge=LEFT).set_z_index(9)
        self.at("zero")
        self.sfx("lock", t=("zero", 0.1))
        self.put(r_zero, rt=0.4, sfx=False)
        self.at("straight")
        self.play(Indicate(r_zero, color=CYAN, scale_factor=1.15), run_time=0.4)
        self.at("straight", 0.5)
        self.play(self.ping(TGT), run_time=0.5)

        # ---------------------------------------------------------------- Walleye : verrouille avant le largage
        self.at("walleye", -0.6)
        self.drop(FadeOut(bomb), FadeOut(F1.trail), mon.vis.animate.set_value(0), sight.op.animate.set_value(0),
                  rt=0.5)
        self.fl = None
        bomb.update(0)
        mon.lk.set_value(0)
        mon.lkp.set_value(0.45)
        r_w, r_c = self.row("WALLEYE", 0, 28), self.row("CONTRAST", 1, 26)
        r_nh, r_ff = self.row("NO HUMAN", 2, 26), self.row("FIRE AND FORGET", 3, 26)
        self.at("walleye")
        self.put(r_w, anims=[FadeIn(bomb), sight.op.animate.set_value(0.6)], rt=0.4)
        self.at("contrast")
        self.put(r_c, anims=[mon.vis.animate.set_value(1)], rt=0.4)
        self.at("contrast", 1.0)
        self.sfx("lock", t=("contrast", 1.25))
        self.play(mon.lk.animate.set_value(1), mon.lkp.animate.set_value(0), run_time=0.4, sfx=False)
        self.at("nohuman", -1.0)
        F2 = self.flight(P_WALLEYE, ("nohuman", -1.0), end="fnf", clack=True, gain=0.3)
        self.at("nohuman")
        self.put(r_nh, rt=0.4)
        self.at("fnf", -1.05)
        self.put(r_ff, rt=0.4)
        self.at("fnf")
        self.play(self.ping(TGT), run_time=0.5)

        # ---------------------------------------------------------------- GBU-15 : un homme dans la boucle
        self.at("gbu", -0.7)
        self.drop(FadeOut(bomb), FadeOut(F2.trail), mon.vis.animate.set_value(0), sight.op.animate.set_value(0),
                  rt=0.4)
        self.fl = None
        bomb.update(0)
        mon.lk.set_value(0)
        mon.lkp.set_value(0.45)
        r_g, r_dl = self.row("GBU-15", 0, 28), self.row("DATA LINK", 1, 26)
        r_op, r_mil = self.row("OPERATOR", 2, 26), self.row("MAN IN THE LOOP", 3, 24)
        self.at("gbu")
        self.put(r_g, anims=[FadeIn(bomb)], rt=0.4)
        self.at("gbu", 0.6)
        F3 = self.flight(P_GBU, ("gbu", 0.6), end=("light", -1.4), clack=True, gain=0.3)
        self.at("datalink")
        self.sfx("tick", t=("datalink", 0.2))
        self.put(r_dl, anims=[mini.vis.animate.set_value(1), mini.tg.animate.set_value(1),
                              mini.ct.animate.set_value(1), link.a.animate.set_value(1)], rt=0.5)
        self.at("operator")
        err2.glide(0.0, "operator", ("mil", 0.6), sound=False)
        self.sfx("lock", t=("operator", 0.25))
        self.put(r_op, anims=[mini.lk.animate.set_value(1), mini.lkp.animate.set_value(0)], rt=0.4, sfx=False)
        self.at("mil")
        self.put(r_mil, anims=[link.b.animate.set_value(1)], rt=0.5)
        self.at("light", -1.4)
        self.play(self.ping(TGT), run_time=0.4)
        self.at("light", -0.9)
        self.drop(FadeOut(bomb), FadeOut(F3.trail), mini.vis.animate.set_value(0), link.a.animate.set_value(0),
                  link.b.animate.set_value(0), rt=0.5)
        self.fl = None
        bomb.update(0)

        # ---------------------------------------------------------------- la camera perd la cible
        r_l, r_n = self.row("LIGHT", 0, 26), self.row("NIGHT", 1, 26)
        r_s, r_f = self.row("SMOKE", 2, 26), self.row("FOG", 3, 26)
        mon.lk.set_value(1)
        mon.lkp.set_value(0)
        self.at("light")
        self.put(r_l, anims=[FadeIn(bomb), mon.vis.animate.set_value(1), sight.op.animate.set_value(0.6)], rt=0.4)
        self.at("light", 0.5)
        t0, t1 = self.T(("light", 0.5)), self.T(("fn1", -1.25))
        F4 = self.flight(lost_path(self.lost_frac(t0, t1)), ("light", 0.5), end=("fn1", -1.25), clack=True,
                         gain=0.3, thud=0.6)
        self.at("night")
        mon.nz.glide(0.3, "night", ("night", 0.6), sound=False)
        self.sfx("crackle", t="night", dur=0.7, density=60, gain=0.3)
        self.put(r_n, rt=0.35)
        self.at("smoke")
        mon.nz.glide(0.65, "smoke", ("smoke", 0.6), sound=False)
        self.sfx("crackle", t="smoke", dur=0.7, density=100, gain=0.35)
        self.put(r_s, rt=0.35)
        self.at("fog")
        mon.nz.glide(1.0, "fog", ("fog", 0.9), sound=False)
        self.sfx("crackle", t="fog", dur=1.4, density=150, gain=0.4)
        self.put(r_f, rt=0.35)
        self.at("lost")
        mon.det.glide(1.0, "lost", ("lost", 0.9), sound=False)
        self.sfx("clack", t="lost", gain=0.5)
        self.play(mon.q.animate.set_value(1), run_time=0.4, sfx="tick")
        self.at("fn1", -1.25)
        gap = ground_cote(X_MISS, XT - BW / 2, GY)
        self.play(FadeIn(gap), run_time=0.35, sfx=False)
        self.at("fn1", -0.8)
        self.drop(FadeOut(gap), FadeOut(bomb), FadeOut(F4.trail), FadeOut(air), mon.vis.animate.set_value(0),
                  sight.op.animate.set_value(0), rt=0.45)
        self.fl = None

        # ---------------------------------------------------------------- GUIDANCE = f( IMAGE )
        toks = [TX(s, 34, INK) for s in ["GUIDANCE", "=", "f(", "IMAGE", ")"]]
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

        # ---------------------------------------------------------------- suite
        n1 = TX("NEXT: GUIDANCE THAT SEES", 28, CYAN)
        n2 = TX("HEAT", 28, CYAN)
        nxt = VGroup(n1, n2).arrange(RIGHT, buff=0.24).move_to([0, -0.4, 0]).set_z_index(9)
        self.at("nxt")
        self.show(n1.move_to(n1.get_center()), rt=0.5)
        self.at("loop", -1.0)
        self.play(FadeIn(n2), Indicate(n2, color=CYAN, scale_factor=1.25), run_time=0.5)

        # ---------------------------------------------------------------- boucle : retour a l'image 0
        self.at("loop")
        self.sfx("whoosh", dur=0.8, f0=200, f1=900, gain=0.25)
        air.update(0)
        bomb.update(0)
        self.play(FadeOut(eq), FadeOut(box), FadeOut(n1), FadeOut(n2), FadeIn(air), FadeIn(bomb),
                  FadeIn(hook_group), run_time=0.8, sfx=False)
        self.at("end")
