#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests de aruco_dedup_v2.py — geometrie, deduplication, score, tracking,
planification ROI, politique de balayage.  Usage : python3 test_aruco_dedup_v2.py
(code retour 0 si tout passe)."""
import sys

import cv2
import numpy as np

import aruco_dedup_v2 as v2


fails = []
def chk(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + ("" if cond else f"  [{extra}]"))
    if not cond: fails.append(name)

q = np.array([[100,100],[200,100],[200,200],[100,200]], np.float32)

# --- geometrie
chk("poly_area carre 100", abs(v2.poly_area(q)-10000) < 1e-6, v2.poly_area(q))
d = v2.Detection(7, q)
chk("Detection aire",   abs(d.area-10000) < 1e-6, d.area)
chk("Detection centre", (d.cx, d.cy) == (150.0,150.0), (d.cx,d.cy))
chk("Detection bbox",   d.bbox == (100.0,100.0,200.0,200.0), d.bbox)
chk("Detection scale",  abs(d.scale-100) < 1e-6, d.scale)
chk("Detection dtype",  d.corners.dtype == np.float32 and d.corners.flags["C_CONTIGUOUS"])
chk("Detection liste->ok", v2.Detection(1, np.array([[0,0],[2,0],[2,2],[0,2]])).area == 4.0)

half = q.copy(); half[:,0] = [150,250,250,150]
i_, c_ = v2.overlap_metrics(v2.Detection(1,q), v2.Detection(1,half))
chk("IoU recouvrement 1/3", abs(i_-1/3) < 1e-3, i_)
chk("containment 1/2",      abs(c_-0.5) < 1e-3, c_)
outer = np.array([[50,50],[250,50],[250,250],[50,250]], np.float32)
i2, c2 = v2.overlap_metrics(v2.Detection(1,q), v2.Detection(1,outer))
chk("imbrique: IoU 0.25",   abs(i2-0.25) < 1e-3, i2)
chk("imbrique: incl. 1.0",  abs(c2-1.0) < 1e-3, c2)
far = q + 500
chk("disjoints -> 0", v2.overlap_metrics(v2.Detection(1,q), v2.Detection(1,far)) == (0.0,0.0))
chk("api iou compat", abs(v2.iou(q, outer)-0.25) < 1e-3)
chk("api containment compat", abs(v2.containment(q, outer)-1.0) < 1e-3)
chk("quad_sanity carre ~1", v2.quad_sanity(d) > 0.99, v2.quad_sanity(d))
skew = np.array([[0,0],[200,0],[210,8],[0,8]], np.float32)
chk("quad_sanity aplati bas", v2.quad_sanity(v2.Detection(1,skew)) < 0.3)

# --- dedup
g = np.full((480,640), 240, np.uint8)
m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 100)
g[100:200, 100:200] = m
sc = v2.QualityScorer(4,1)
fn = lambda x: sc(g, x)

k,r = v2.deduplicate([], score_fn=fn);                     chk("vide", (k,r)==([],[]))
k,r = v2.deduplicate([v2.Detection(7,q)], score_fn=fn);    chk("un seul", len(k)==1 and not r)
k,r = v2.deduplicate([v2.Detection(7,q), v2.Detection(7,outer)], score_fn=fn)
chk("imbrique -> interieur garde", len(k)==1 and abs(k[0].area-10000)<1, [x.area for x in k])
chk("motif imbrication", "imbriqu" in r[0][1], r[0][1])
k,r = v2.deduplicate([v2.Detection(7,q), v2.Detection(7,q+3)], score_fn=fn)
chk("meme ID decale -> 1", len(k)==1)
k,r = v2.deduplicate([v2.Detection(7,q), v2.Detection(3,q+300)], score_fn=fn)
chk("2 IDs eloignes -> 2", len(k)==2 and not r)
k,r = v2.deduplicate([v2.Detection(7,q), v2.Detection(3,q+110)], score_fn=fn)
chk("2 IDs adjacents -> 2", len(k)==2 and not r, [x.marker_id for x in k])
k,r = v2.deduplicate([v2.Detection(7,q), v2.Detection(7,q+400)], score_fn=fn)
chk("meme ID loin -> 2 gardes", len(k)==2)
k,r = v2.deduplicate([v2.Detection(7,q), v2.Detection(7,q+400)], score_fn=fn, unique_ids=True)
chk("unique_ids -> 1", len(k)==1)
st={}
k,r = v2.deduplicate([v2.Detection(7,q), v2.Detection(3,q+300)], score_fn=fn, stats=st)
chk("scoring paresseux (0 warp)", st.get("scored",0)==0, st)
st={}
k,r = v2.deduplicate([v2.Detection(7,q), v2.Detection(7,outer)], score_fn=fn, stats=st)
chk("scoring en conflit", st.get("scored",0)==2, st)
tiny = np.array([[140,140],[160,140],[160,160],[140,160]], np.float32)
k,r = v2.deduplicate([v2.Detection(7,q), v2.Detection(3,tiny)], score_fn=fn)
chk("petit marqueur dans grand -> 2", len(k)==2, [x.marker_id for x in k])

# --- qualite
good = v2.Detection(7, np.array([[100,100],[200,100],[200,200],[100,200]], np.float32))
bad  = v2.Detection(7, np.array([[300,300],[400,300],[400,400],[300,400]], np.float32))
chk("score vrai marqueur > faux", sc(g,good) > sc(g,bad) + 0.2, (sc(g,good), sc(g,bad)))
chk("marker_quality compat", abs(v2.marker_quality(g, good.corners, 4) - sc(g,good)) < 1e-6)

# --- tracking
tr = v2.TemporalStabilizer(min_hits=2, max_miss=2)
t0 = 1000.0
out = tr.update([v2.Detection(7,q)], t0);          chk("track 1re image non publiee", len(out)==0)
out = tr.update([v2.Detection(7,q+5)], t0+0.05);   chk("track 2e image publiee", len(out)==1)
t0b = tr.tracks[0]
chk("vitesse estimee", t0b.vx > 1.0 and t0b.vy > 1.0, (t0b.vx,t0b.vy))
chk("prediction en avant", t0b.predict(0.05)[0] > t0b.cx)
tt = t0+0.05
for _ in range(3):
    tt += 0.05
    tr.update([], tt)
chk("track expire", len(tr.tracks)==0)
chk("confiance croissante", v2.Track(v2.Detection(7,q), 0.0).confidence < 1.0)

# dt reel (pas suppose = 1 image) : deux images sautees doivent doubler
# le deplacement predit par rapport a deux images consecutives
tr3 = v2.TemporalStabilizer(min_hits=1, max_miss=5)
tr3.update([v2.Detection(7,q)], 0.0)
tr3.update([v2.Detection(7,q+np.array([10,0],np.float32))], 0.10)  # vx = 100 px/s
tr3.tracks[0]
pred_1step = tr3.tracks[0].predict(0.10)[0]
pred_3steps = tr3.tracks[0].predict(0.30)[0]
chk("dt reel: prediction proportionnelle au dt", abs((pred_3steps-tr3.tracks[0].cx) - 3*(pred_1step-tr3.tracks[0].cx)) < 1e-6)

# ROI planner
pl = v2.RoiPlanner(max_coverage=0.35)
tr2 = v2.TemporalStabilizer(); tr2.update([v2.Detection(7,q)], 0.0)
chk("roi produite", len(pl.plan(tr2.tracks, 1280, 720, expected_dt=1/30))==1)
chk("roi trop large -> vide", pl.plan(tr2.tracks, 200, 200, expected_dt=1/30)==[])
chk("roi bornee a l'image", all(0<=b[0] and b[2]<=1280 for b in pl.plan(tr2.tracks,1280,720,expected_dt=1/30)))
chk("roi elargie plus grande", (lambda a,b: (a[0][2]-a[0][0]) < (b[0][2]-b[0][0]))(
    pl.plan(tr2.tracks,1280,720,expected_dt=1/30), pl.plan(tr2.tracks,1280,720,expected_dt=1/30,margin_scale=1.7)))

# --- ScanPolicy
sp = v2.ScanPolicy(base=4, max_interval=32)
chk("1er appel force complet", sp.want_full([], g)[0])
sp.after_full(1,1); chk("intervalle double", sp.interval >= 4)
sp.after_roi(False); chk("perte -> force complet", sp.force)

print("\nTOUS LES TESTS OK" if not fails else f"\n{len(fails)} ECHEC(S): {fails}")
sys.exit(1 if fails else 0)
