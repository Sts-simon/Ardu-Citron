"""
Bruitages de l'episode 1 (version courte) - charte "Blueprint explicatif".

  python3 guidage_ep01_sfx.py          ->  ep01_sfx.wav (44,1 kHz, stereo, 16 bits)

Les instants viennent du dictionnaire C de guidage_ep01_court.py : apres avoir
recale C sur la voix, relance simplement ce script pour regenerer le son.

Palette sonore (tres discrete par rapport a la voix) :
  - tics en gamme pentatonique sur chaque apparition
  - souffle filtre pendant les vols et les deplacements de l'avion
  - petits "clacks" mecaniques aux largages, thuds sourds aux impacts
  - craquements pour la rafale
  - montee puis impact grave sur le mot final (RELEASE)
"""
import os
import sys
import wave

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guidage_ep01_court as ep  # noqa: E402  (C, END_T, GUST_IDX, P_GUST)

SR = 44100
C, END_T = ep.C, ep.END_T
TAIL = 2.5
N = int((END_T + TAIL) * SR)
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


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


# ------------------------------------------------------------------ briques sonores
def pluck(freq, dur=0.55, tau=0.11):
    t = tvec(dur)
    s = (np.sin(2 * np.pi * freq * t) + 0.30 * np.sin(2 * np.pi * 2 * freq * t)
         + 0.10 * np.sin(2 * np.pi * 3 * freq * t))
    e = np.exp(-t / tau)
    a = int(0.003 * SR)
    e[:a] *= np.linspace(0, 1, a)
    return s * e * 0.5


def noise_lp(cut):
    """Bruit blanc passe dans un passe-bas 2 poles dont la coupure varie (Hz, un point par echantillon)."""
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


def whoosh(dur, f0, f1):
    u = np.linspace(0, 1, int(dur * SR))
    cut = f0 * (f1 / f0) ** np.sin(np.pi * u)           # monte puis redescend : un souffle
    env = np.sin(np.pi * u) ** 2
    return noise_lp(cut) * env


def thud(f0=95.0, f1=42.0, dur=0.55, click=0.25):
    t = tvec(dur)
    f = f1 + (f0 - f1) * np.exp(-t / 0.06)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18)
    c = rng.standard_normal(len(t)) * np.exp(-t / 0.008) * click
    return s + c


def clack():
    t = tvec(0.09)
    n = rng.standard_normal(len(t)) * np.exp(-t / 0.010)
    n = np.diff(n, prepend=0.0)                          # un peu de brillance
    lo = np.sin(2 * np.pi * 150 * t) * np.exp(-t / 0.025)
    return 0.6 * n + 0.8 * lo


def crackle(dur, density=70):
    s = np.zeros(int(dur * SR))
    for _ in range(int(density * dur)):
        i = rng.integers(0, len(s) - 200)
        k = np.arange(200) / SR
        s[i:i + 200] += rng.choice([-1, 1]) * rng.uniform(0.3, 1.0) * np.exp(-k / 0.0015)
    return np.diff(s, prepend=0.0)


def rise(dur):
    u = np.linspace(0, 1, int(dur * SR))
    cut = 300 * (7000 / 300) ** u
    nz = noise_lp(cut) * u ** 2.2
    f = 220 * (1100 / 220) ** u
    sw = np.sin(2 * np.pi * np.cumsum(f) / SR) * u ** 2.2 * 0.25
    return nz + sw


def impact():
    t = tvec(2.2)
    f = 38 + 60 * np.exp(-t / 0.08)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.55)
    burst = noise_lp(np.full(len(t), 900.0)) * np.exp(-t / 0.12) * 0.8
    return sub + burst


# ------------------------------------------------------------------ 1. tics pentatoniques
PENT = [0, 2, 4, 7, 9, 12, 14, 16]
base = 523.25  # C5


def note_walk(keys):
    idx = 2
    for n, k in enumerate(keys):
        step = int(rng.choice([-2, -1, 1, 1, 2]))
        idx = idx + step
        if idx < 0 or idx >= len(PENT):
            idx = max(0, min(len(PENT) - 1, idx - 2 * step))
        yield k, base * 2 ** (PENT[idx] / 12), (-0.3 if n % 2 else 0.3)


tick_keys = ["hook2", "no_comp", "no_sens", "no_steer", "physics", "g_arrow", "v_arrow", "ballistic",
             "drag", "wind", "gap", "error", "grows", "aircraft", "computer", "in_speed", "in_alt",
             "in_wind", "solves", "ccip", "impact_pt", "now", "target", "ccrp", "ccrp_flight",
             "ccrp_target", "ccrp_mark", "ccrp_label", "rocket", "no_steer2", "pred_rocket",
             "prediction", "release_word", "no_corr", "eq1", "eq2", "eq3", "eq4", "eq5", "nxt"]
for k, f, pan in note_walk(tick_keys):
    add("tick", pluck(f), C[k], pan, 1.0)

# ping du marqueur ("shows") : tic aigu plus long
add("tick", pluck(base * 4, dur=0.9, tau=0.28), C["shows"], 0.0, 0.8)
# cible verrouillee : petit intervalle quinte
add("tick", pluck(base * 1.5), C["put"], -0.2, 0.9)
add("tick", pluck(base * 2.0), C["put"] + 0.09, 0.2, 0.9)

# ------------------------------------------------------------------ 2. largages (clack) et impacts (thud)
for k in ("release", "rel_ccip", "ccrp_rel", "fire", "go"):
    add("hit", clack(), C[k], 0.0, 0.55)

t_land_v = C["land"]
t_land_d = C["drag"] + 1.0
t_land_w = C["wind"] + 1.2
t_land_f1 = C["rel_ccip"] + 1.3
t_land_f2 = C["ccrp_rel"] + 1.3
t_land_r = C["fire"] + 1.1
t_land_g = C["go"] + 2.2
for t, g in ((t_land_v, 0.9), (t_land_d, 0.5), (t_land_w, 0.7), (t_land_f1, 0.9),
             (t_land_f2, 0.9), (t_land_r, 0.8), (t_land_g, 0.8)):
    add("hit", thud(), t, 0.0, g * 0.9)
# la roquette claque un peu plus sec a l'arrivee
add("hit", crackle(0.25, 120), t_land_r, 0.0, 0.25)

# ------------------------------------------------------------------ 3. souffle pendant les vols / deplacements
moves = [
    (C["release"], C["land"] - C["release"], 250, 1400, 0.60, 0.0),   # chute libre
    (C["drag"], 1.0, 300, 1600, 0.50, 0.0),
    (C["wind"], 1.2, 400, 2400, 0.60, 0.0),
    (C["to_air"] + 0.5, 0.7, 200, 900, 0.30, 0.0),                    # l'avion se place
    (C["back"], 0.9, 200, 900, 0.30, 0.0),
    (C["move"], C["put"] - C["move"], 200, 1000, 0.35, 0.0),           # le reticule glisse sur la cible
    (C["rel_ccip"], 1.3, 300, 1500, 0.55, 0.0),
    (C["rel_ccip"] + 1.4, 0.7, 200, 900, 0.30, 0.0),
    (C["fly"], C["ccrp_rel"] - C["fly"], 200, 900, 0.35, 0.0),         # approche CCRP
    (C["ccrp_rel"], 1.3, 300, 1500, 0.55, 0.0),
    (C["ccrp_rel"] + 1.4, 0.8, 200, 900, 0.30, 0.0),
    (C["fire"], 1.1, 700, 3800, 0.60, 0.0),                           # roquette : plus brillant
    (C["fire"] + 1.2, 0.6, 200, 900, 0.30, 0.0),
    (C["go"], 2.2, 250, 1300, 0.60, 0.0),
    (C["loop"], 0.8, 200, 900, 0.25, 0.0),
]
for t0, d, f0, f1, g, pan in moves:
    add("move", whoosh(d, f0, f1), t0, pan, g)

# ------------------------------------------------------------------ 4. craquements de la rafale
tg = C["go"] + 2.2 * ep.GUST_IDX / (len(ep.P_GUST) - 1)
add("move", crackle(0.7, 110), tg - 0.25, 0.0, 0.5)
# vent : petit souffle plus aigu sous la ligne WIND
add("move", whoosh(1.2, 900, 3200), C["wind"], 0.2, 0.15)

# ------------------------------------------------------------------ 5. montee + impact grave sur RELEASE
rise_len = C["eq_box"] - C["eq1"]
add("final", rise(rise_len), C["eq1"], 0.0, 0.55)
add("final", impact(), C["eq_box"], 0.0, 1.0)

# ------------------------------------------------------------------ mixage
# reverb courte sur les tics, plus longue sur la montee/impact
ir_s = rng.standard_normal(int(0.45 * SR)) * np.exp(-np.arange(int(0.45 * SR)) / (0.12 * SR))
ir_l = rng.standard_normal(int(1.8 * SR)) * np.exp(-np.arange(int(1.8 * SR)) / (0.5 * SR))


def wet(x, ir, amount):
    out = np.zeros_like(x)
    for c in (0, 1):
        out[:, c] = fftconvolve(x[:, c], ir / np.sqrt(np.sum(ir ** 2)))[:N] * amount
    return out


mix = (bus["tick"] * 0.26 + wet(bus["tick"], ir_s, 0.10)
       + bus["move"] * 0.12
       + bus["hit"] * 0.30
       + bus["final"] * 0.60 + wet(bus["final"], ir_l, 0.12))

# fondu en entree/sortie, boucle propre
fade = int(0.05 * SR)
mix[:fade] *= np.linspace(0, 1, fade)[:, None]
end = int(END_T * SR)
mix[end:] *= np.exp(-np.arange(N - end) / (0.35 * SR))[:, None]   # queue d'impact, amortie apres la fin de la video
peak = np.max(np.abs(mix))
mix = mix / peak * 0.8

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ep01_sfx.wav")
pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
with wave.open(path, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f"OK : {path}  ({N / SR:.1f} s, pic 0.8)")
