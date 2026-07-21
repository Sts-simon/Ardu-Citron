import cv2
import numpy as np
import re


INPUT = "markers.h"
OUTPUT = "mon_dictionnaire_aruco.npy"


# ==========================
# Lecture markers.h
# ==========================

with open(INPUT, "r") as f:
    data = f.read()


codes = re.findall(
    r"0b([01]{16})",
    data
)


print("Nombre de marqueurs :", len(codes))


# ==========================
# Conversion bits -> matrice 4x4
# ==========================

markers = []


for code in codes:

    bits = np.array(
        [int(b) for b in code],
        dtype=np.uint8
    )


    marker = bits.reshape(
        (4,4)
    )


    markers.append(marker)



markers=np.array(
    markers,
    dtype=np.uint8
)


print("Premier marqueur :")
print(markers[0])



# ==========================
# Conversion OpenCV
# ==========================

byte_list=[]


for marker in markers:

    b=cv2.aruco.Dictionary_getByteListFromBits(
        marker
    )

    byte_list.append(b)



bytesList=np.concatenate(
    byte_list,
    axis=0
)



print(
    "bytesList shape:",
    bytesList.shape
)



# ==========================
# Création dictionnaire
# ==========================

dictionary=cv2.aruco.Dictionary(
    bytesList,
    4
)


# correction bits
dictionary.maxCorrectionBits = 0



# sauvegarde

np.save(
    OUTPUT,
    dictionary.bytesList
)


print("Dictionnaire sauvegardé :", OUTPUT)
