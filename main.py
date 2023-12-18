import cv2 as cv
import os
import numpy as np
import itertools

def vector(p1, p2):
    return p2 - p1

def distance(v):
    return (v[0]**2 + v[1]**2)**0.5

def get_rad_angle(V1, V2):
    return np.arccos(np.dot(V1, V2) / (distance(V1) * distance(V2)))

def find_vector(h, o):
    x = h * np.cos(o)
    y = h * np.sin(o)
    return np.array([x, y])

path = "Michelangelo/Michelangelo/frag_eroded"

img2 = cv.imread('Michelangelo/Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg')

# Paramètres
MIN_MATCH_COUNT = 10

# Initiate FAST detector
fast = cv.FastFeatureDetector_create()
brisk = cv.BRISK_create()

# Charger les données des fragments et des fragments à ignorer

img_width, img_height = 2000, 1500

# Créer une image vide de la taille de l'image de sortie
image = np.zeros((img_height, img_width, 3), np.uint8)  # Utilisez 3 canaux pour RGB (fond noir, chaque pixel à zéro)

# Placer les fragments dans l'image de sortie avec une marge à partir de (200, 200)
margin_x, margin_y = 200, 200
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
    good = [m for m in matches if m.distance < 80]

    # Dictionnaire pour stocker les paramètres et leurs votes
    parameters = {(0, 0, 0): 0}

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

        # Trouver les paires d'indices uniques
        for pair in itertools.combinations(range(len(src_pts)), 2):
            i, j = pair
            v = vector(src_pts[i], src_pts[j])
            v_center = vector(src_pts[i], center)
            v_image = vector(dst_pts[i], dst_pts[j])

            # Replacer le centre du fragment dans l'image
            x, y = v_image
            b = np.arctan2(y, x)
            a = get_rad_angle(v, v_center)
            o = a + b - np.pi / 2
            v_center_image = -find_vector(distance(v_center), o)

            # Calculer les angles
            theta = get_rad_angle(v_center, v_center_image)

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
                        try:
                            parameters[(theta, tx, ty)] += 1
                        except:
                            parameters[(theta, tx, ty)] = 1

        max_params = max(parameters, key=parameters.get)
        print("Parameters with the most votes:", np.degrees(max_params[0]), max_params[1:])
        angle, txr, tyr = max_params
        to_write.append(f"{index} {round(txr)} {round(tyr)} {np.degrees(angle):.4f}\n")

# Écrire les résultats dans un fichier texte
with open('results.txt', 'w') as file:
    file.writelines(to_write)
# crop l'image 1775x775
crop_img = image[200:975, 200:1907]

# Afficher l'image de sortie
cv.imwrite("output.png", crop_img)
cv.imshow("Image Finale", crop_img)
cv.waitKey(0)
cv.destroyAllWindows()