import cadquery as cq
from cadquery import exporters
import math

# ============================================================
# 1. PARAMÈTRES (Fusionnés et placés avant leur utilisation)
# ============================================================
CLR_R = 0.15   # jeu radial standard FDM (+0.3mm diametral)

# --- Corps (tube jaune / porte-oculaire) ---
Y_SHAFT_START   = 0.0
Y_SHAFT_END     = 65.0
R_SHAFT_OUT     = 18.0      # diam 36mm
R_BORE          = 12.5      # alesage central diam 25mm (constant)

Y_FLANGE_START  = -1.2
Y_FLANGE_END    = 0.0
R_FLANGE        = 20.0      # collerette / bague : diam 40mm

Y_NECK_START    = -2.2
Y_NECK_END      = -1.2
R_NECK          = 14.55     # diam 29.1mm

# --- Corps001 (cylindre vert / oculaire) ---
Y_OCU_START     = -39.3
Y_OCU_END       = -2.2
R_OCU           = 15.0      # diam 30mm (fut principal de l'oculaire)

# --- Caméra (mesures réelles) ---
CAM_HOLE_X      = 10.5      # mesure precise
CAM_HOLE_Z_ROW1 = 0.0       
CAM_HOLE_Z_ROW2 = -12.5  
CAM_HOLE_D      = 2.4

Y_CAM_FRONT   = -47.0   
Y_CAM_BACK    = -49.12  
Y_LENS_TIP    = -40.47  
R_ACTUATOR    = 7.64    

# --- Conception de la pièce ---
PLATE_THK      = Y_CAM_FRONT - (-44.0)
Y_PLATE_START  = Y_CAM_FRONT             
Y_PLATE_END    = -44.0

R_LENS_BORE    = 10.0     

# -- Fourreau fermé autour de l'oculaire --
R_SLEEVE_IN   = R_OCU + CLR_R      # 15.15
R_SLEEVE_OUT  = 19.0

# -- Zone pince --
Y_FINGER_ROOT = -20.0    # les fentes remontent sur 17.8mm
Y_FINGER_TIP  = 10.0     # recouvrement sur le porte-oculaire

N_SLOTS   = 6
SLOT_W    = 1.3
N_FINGERS = 4  # nombre de petales 

# -- Ouverture rectangulaire --
RECT_OPENING_W   = 13.0     
RECT_OPENING_H   = 19.0     
RECT_OPENING_ZC  = -3.7     

# -- Verrouillage à baïonnette --
FLARE_Y0       = 7.5    
FLARE_Y1       = 9.0    
FLARE_R        = 23.5   
RAMP_ANGLE     = 45.0   
SLOT_CLR       = 0.5    

# ============================================================
# 2. FONCTIONS DE BASE
# ============================================================
YPLANE = cq.Plane(origin=(0, 0, 0), xDir=(1, 0, 0), normal=(0, 1, 0))

def cyl_along_y(radius, y0, y1, cx=0.0, cz=0.0):
    h = y1 - y0
    return (
        cq.Workplane(YPLANE)
        .workplane(offset=y0)
        .center(cx, cz)
        .circle(radius)
        .extrude(h)
    )

def box_along_y(w, h, y0, y1, cx=0.0, cz=0.0):
    dy = y1 - y0
    return (
        cq.Workplane(YPLANE)
        .workplane(offset=y0)
        .center(cx, cz)
        .rect(w, h, centered=True)
        .extrude(dy)
    )

def poly_solid(pts):
    pts = [(float(r), float(y)) for r, y in pts]
    wp = cq.Workplane("XY").moveTo(*pts[0])
    for p in pts[1:]:
        wp = wp.lineTo(*p)
    last_r, last_y = pts[-1]
    first_r, first_y = pts[0]
    if last_r > 1e-6:
        wp = wp.lineTo(0.0, last_y)
    if first_r > 1e-6:
        wp = wp.lineTo(0.0, first_y)
    wp = wp.close()
    return wp.revolve(360, (0, 0, 0), (0, 1, 0))

# ============================================================
# 3. CONSTRUCTION DU PROFIL
# ============================================================
inner_pts = [
    (R_SLEEVE_IN,                 -39.3),                 
    (R_SLEEVE_IN,                 Y_FINGER_ROOT),         
    (R_NECK + CLR_R,              -2.2),
    (R_NECK + CLR_R,              -1.8),
    (R_FLANGE + 0.3,              -1.2),
    (R_FLANGE + 0.3,               0.0),
    (R_SHAFT_OUT - 0.4,            0.9),
    (R_SHAFT_OUT - 0.4,            2.0),
    (R_SHAFT_OUT + CLR_R,          3.0),
    (R_SHAFT_OUT + CLR_R,          Y_FINGER_TIP),         
]

outer_pts = [
    (21.0,                        Y_PLATE_START),         
    (21.0,                        Y_PLATE_END),             
    (R_SLEEVE_OUT,                -39.3),                   
    (R_SLEEVE_OUT,                Y_FINGER_ROOT),           
    (16.7,                        -18.0),                   
    (16.7,                        -5.0),                    
    (23.0,                        -1.2),                    
    (23.0,                         0.0),                      
    (20.5,                         0.9),                      
    (20.5,                         FLARE_Y0),                 
    (FLARE_R,                      FLARE_Y1),                
    (FLARE_R,                      Y_FINGER_TIP),           
]

print("Construction des solides de revolution...")
outer_solid = poly_solid(outer_pts)
inner_solid = poly_solid(inner_pts)

# --- Ouverture rectangulaire ---
rect_cut = box_along_y(RECT_OPENING_W, RECT_OPENING_H, Y_PLATE_START - 0.5, -38.0, cx=0.0, cz=RECT_OPENING_ZC)

inner_total = inner_solid.union(rect_cut)
body = outer_solid.cut(inner_total)
print("  -> volume coque =", round(body.val().Volume(), 1), "mm3")

# --- 4 trous de fixation M2 camera ---
for zc in (CAM_HOLE_Z_ROW1, CAM_HOLE_Z_ROW2):
    for xs in (1, -1):
        h = cyl_along_y(CAM_HOLE_D/2, Y_PLATE_START-1, Y_PLATE_END+1, cx=xs*CAM_HOLE_X, cz=zc)
        body = body.cut(h)
print("4 trous camera decoupes")

# --- Fentes des 4 pétales ---
Y0 = Y_FINGER_ROOT - 0.01
Y1 = Y_FINGER_TIP + 0.5
R_CUT = R_SHAFT_OUT + 10.0
FINGER_ARC = 45.0   
GAP_ARC = 360.0/N_FINGERS - FINGER_ARC   

def wedge_cut(y0, y1, start_deg, end_deg, r_cut):
    a0 = math.radians(start_deg)
    a1 = math.radians(end_deg)
    p0 = (r_cut*math.cos(a0), r_cut*math.sin(a0))
    p1 = (r_cut*math.cos(a1), r_cut*math.sin(a1))
    return (
        cq.Workplane(YPLANE)
        .workplane(offset=y0)
        .moveTo(0, 0)
        .lineTo(*p0)
        .radiusArc(p1, r_cut)
        .close()
        .extrude(y1 - y0)
    )

for i in range(N_FINGERS):
    center = (360.0/N_FINGERS)*i + (360.0/N_FINGERS)/2.0   
    wedge = wedge_cut(Y0, Y1, center - GAP_ARC/2.0, center + GAP_ARC/2.0, R_CUT)
    body = body.cut(wedge)

print("4 petales decoupes. Volume final =", round(body.val().Volume(), 1), "mm3")

# ============================================================
# 4. EXPORT
# ============================================================
# Les chemins sont modifiés pour exporter dans le dossier courant
exporters.export(body, 'PartA_collet_camera.step')
exporters.export(body, 'PartA_collet_camera.stl', tolerance=0.05, angularTolerance=0.2)
print("Part A exportee avec succès (STEP + STL) dans le dossier courant.")
