import cv2 as cv
import os
import numpy as np
import itertools
import functools

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

    asso = [ [np.array(kp1[m.queryIdx].pt), np.array(kp2[m.trainIdx].pt)] for m in matches if m.distance < 50 ]
    if len(asso) == 0:
        continue
    couples={}
    for pair in itertools.combinations(range(len(asso)), 2):
        i,j = pair
        src1,dst1=asso[i]
        src2,dst2=asso[j]
        d_src=distance(vector(src1,src2))
        d_dst=distance(vector(dst1,dst2))

        if d_src == d_dst:
            try:
                couples[i].append(j)
            except:
                couples[i]=[j]
            try:
                couples[j].append(i)
            except:
                couples[j]=[i]
    if len(couples) == 0:
        continue
    m=max(couples.items(), key=lambda x:len(x[1]))
    if len(m[1]) < 2:
        continue

    good=[ asso[m[0]], asso[m[1][0]] ]
    
    src=[good[0][0],good[1][0]]
    dst=[good[0][1],good[1][1]]
        
    x1, y1 = vector(src[0], src[1])
    x2, y2 = vector(dst[0], dst[1])

    # Calculer les angles
    theta1 = np.arctan2(y1, x1)
    theta2 = np.arctan2(y2, x2)
    theta = theta2 - theta1

    # Calculer tx et ty
    tx = dst[0][0] - src[0][0] * np.cos(theta) + src[0][1] * np.sin(theta)
    ty = dst[0][1] - src[0][0] * np.sin(theta) - src[0][1] * np.cos(theta)
        
    print("Parameters :",f"{index} {round(tx)} {round(ty)} {np.degrees(theta):.4f}")
    to_write.append(f"{index} {round(tx)} {round(ty)} {np.degrees(theta):.4f}\n")

# Écrire les résultats dans un fichier texte
with open('results2.txt', 'w') as file:
    file.writelines(to_write)
print('\nFILE GENERATED')