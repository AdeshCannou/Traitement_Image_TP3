import cv2 as cv
import os
import numpy as np
import itertools

def vector(p1, p2):
    return p2 - p1

def distance(v):
    return (v[0]**2 + v[1]**2)**0.5

path = "Michelangelo/Michelangelo/frag_eroded"

img2 = cv.imread('Michelangelo/Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg')

# Paramètres
MIN_MATCH_COUNT = 10

# Initiate FAST detector
fast = cv.FastFeatureDetector_create()
brisk = cv.BRISK_create()

to_write=[]
for index in range(328):
    fragment_filename = os.path.join(path, f"frag_eroded_{int(index)}.png")

    # Charger les fragments
    img1 = cv.imread(fragment_filename)
    center = (img1.shape[1] // 2, img1.shape[0] // 2)
        
    # Trouver les keypoints et les descriptors avec SIFT
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    kp1, des1 = brisk.compute(img1, kp1)
    kp2, des2 = brisk.compute(img2, kp2)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = [m for m in matches if m.distance < 50]

    # Dictionnaire pour stocker les paramètres et leurs votes
    parameters = {}

    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

        # Trouver les paires d'indices uniques
        for pair in itertools.combinations(range(len(src_pts)), 2):
            i, j = pair
            x1, y1 = vector(src_pts[i], src_pts[j])
            x2, y2 = vector(dst_pts[i], dst_pts[j])

            # Calculer les angles
            theta1 = np.arctan2(y1, x1)
            theta2 = np.arctan2(y2, x2)
            theta = theta2 - theta1

            # Calculer tx et ty
            tx = dst_pts[i][0] - src_pts[i][0] * np.cos(theta) + src_pts[i][1] * np.sin(theta)
            ty = dst_pts[i][1] - src_pts[i][0] * np.sin(theta) - src_pts[i][1] * np.cos(theta)

            # Vérifier les autres associations
            for k in range(len(src_pts)):
                if k not in [i, j]:
                    x_prime = src_pts[k][0] * np.cos(theta) - src_pts[k][1] * np.sin(theta) + tx
                    y_prime = src_pts[k][0] * np.sin(theta) + src_pts[k][1] * np.cos(theta) + ty

                    distance_x = np.abs(x_prime - dst_pts[k][0])
                    distance_y = np.abs(y_prime - dst_pts[k][1])

                    if distance_x < 1.0 and distance_y < 1.0:
                        parameters[(theta, tx, ty)] = parameters.get((theta, tx, ty), 0) + 1
        try:
            max_params = max(parameters, key=parameters.get)
            print(index, "Parameters with the most votes:", np.degrees(max_params[0]), max_params[1:])
            angle, txr, tyr = max_params
            to_write.append(f"{index} {round(txr)} {round(tyr)} {np.degrees(angle):.4f}\n")
        except:
            print('No valid parameters')

# Écrire les résultats dans un fichier texte
with open('results.txt', 'w') as file:
    file.writelines(to_write)
print('\nFILE GENERATED')