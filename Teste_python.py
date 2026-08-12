import cv2
import numpy as np
from astropy.wcs import WCS

def encadrer_avec_wcs(img_grand_champ_path, crop_path, wcs_large_fits, wcs_crop_fits, output_path="11_encadre.jpg"):
    # 1. Charger l'image principale
    img = cv2.imread(img_grand_champ_path)
    crop = cv2.imread(crop_path)
    if img is None or crop is None:
        print("Erreur de chargement des images.")
        return

    # 2. Charger les solutions WCS
    wcs_large = WCS(wcs_large_fits)
    wcs_crop = WCS(wcs_crop_fits)

    h_crop, w_crop = crop.shape[:2]

    # 3. Récupérer les 4 coins du crop en pixels
    # Astropy utilise la convention 0-indexée avec origin=0
    coins_pixels_crop = np.array([
        [0, 0],
        [w_crop - 1, 0],
        [w_crop - 1, h_crop - 1],
        [0, h_crop - 1]
    ], dtype=np.float64)

    # 4. Convertir les pixels du crop -> Coordonnées célestes (RA, Dec en degrés)
    sky_coords = wcs_crop.pixel_to_world_values(coins_pixels_crop)

    # 5. Convertir les coordonnées célestes -> Pixels sur l'image grand champ
    pixels_large = wcs_large.world_to_pixel_values(sky_coords)
    pts_int = np.int32(pixels_large).reshape((-1, 1, 2))

    print("Coordonnées calculées sur l'image 1 :")
    for i, pt in enumerate(pts_int[:, 0]):
        print(f"  Coin {i+1} : X={pt[0]}, Y={pt[1]}")

    # 6. Dessiner le cadre rouge et un repère
    epaisseur = max(4, int(min(img.shape[:2]) / 250))
    cv2.polylines(img, [pts_int], isClosed=True, color=(0, 0, 255), thickness=epaisseur)

    # Centre du rectangle
    cx = int(np.mean(pts_int[:, 0, 0]))
    cy = int(np.mean(pts_int[:, 0, 1]))
    rayon = max(25, epaisseur * 5)

    cv2.imwrite(output_path, img)
    print(f"\n Image générée avec succès : {output_path}")

if __name__ == "__main__":
    encadrer_avec_wcs("11.png", "8.png", "wcs_11.fits", "wcs_8.fits")
