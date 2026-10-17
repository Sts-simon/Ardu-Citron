import numpy as np, wave
SR = 44100
DUR = 102.54
n = int(SR * DUR)
out = np.zeros(n, dtype=np.float64)
rng = np.random.default_rng(11)

def add(t0, sig):
    i = int(t0 * SR)
    j = min(n, i + len(sig))
    if i < n:
        out[i:j] += sig[: j - i]

def tick(f, amp=0.05, d=0.07):
    t = np.arange(int(SR * d)) / SR
    return amp * np.sin(2 * np.pi * f * t) * np.exp(-t * 55) + amp * 0.4 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 90)

def lowpass(x, a):
    y = np.zeros_like(x)
    acc = 0.0
    for k in range(len(x)):
        acc += a * (x[k] - acc)
        y[k] = acc
    return y

def whoosh(t0, d, amp, a0=0.04, a1=0.35):
    m = int(SR * d)
    noise = rng.normal(0, 1, m)
    # filtre dont la frequence de coupure monte au cours du vol
    y = np.zeros(m); acc = 0.0
    for k in range(m):
        a = a0 + (a1 - a0) * (k / m)
        acc += a * (noise[k] - acc)
        y[k] = acc
    env = np.sin(np.pi * np.linspace(0, 1, m)) ** 1.5
    y = y / (np.max(np.abs(y)) + 1e-9) * env * amp
    add(t0, y)

# micro-sons sur les apparitions d'etiquettes / termes (gamme pentatonique, tres discret)
scale = [880, 987.8, 1108.7, 1318.5, 1480]
cues = [3.42, 7.1, 7.5, 9.66, 10.62, 12.24, 13.06, 13.7, 16.04, 16.8, 18.74, 20.3, 22.5, 29.74,
        33.0, 34.76, 35.5, 37.7, 38.4, 39.84, 45.8, 46.5, 47.4, 50.3, 51.9, 54.0, 55.3, 58.05, 58.95,
        60.95, 62.3, 65.5, 67.44, 68.0, 69.0, 76.5, 81.5, 85.0, 85.7, 86.9, 91.2, 92.8, 93.4, 94.1,
        96.36, 96.9, 97.5]
for k, t in enumerate(cues):
    add(t, tick(scale[k % len(scale)] * (1 if k % 7 else 2), amp=0.045))

# vol principal 25.4 -> 29.3, petite correction 62.4 -> 63.5, derive silencieuse
whoosh(25.4, 3.9, 0.10)
whoosh(62.4, 1.1, 0.06)
# bruit de mesure 72.0 -> 74.0 : craquements
for t in np.arange(72.0, 74.0, 0.05):
    if rng.random() < 0.6:
        m = int(SR * 0.012)
        add(t, rng.normal(0, 1, m) * np.exp(-np.linspace(0, 6, m)) * 0.04)
# nuage d'incertitude : petite lueur
t = np.arange(int(SR * 0.5)) / SR
add(76.5, 0.03 * np.sin(2 * np.pi * 220 * t * (1 + 0.5 * t)) * np.exp(-t * 4))
# montee 98.38 -> 99.05 puis impact grave sur ERROR (discret pour ne pas masquer la voix)
m = int(SR * 0.67)
tt = np.arange(m) / SR
f = 300 + 700 * (tt / 0.67)
add(98.38, 0.035 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.linspace(0, 1, m))
t = np.arange(int(SR * 1.0)) / SR
add(99.05, 0.16 * np.sin(2 * np.pi * 58 * t) * np.exp(-t * 4.5))
# fin de boucle : petit "whoosh" de retour 101.7 -> 102.5
whoosh(101.7, 0.8, 0.05)

out = np.clip(out, -0.9, 0.9)
with wave.open("sfx.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((out * 32767).astype(np.int16).tobytes())
print("sfx ok", np.max(np.abs(out)))
