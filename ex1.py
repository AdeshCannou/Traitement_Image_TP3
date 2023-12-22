import cv2
import numpy as np

# Charger les images
img1 = cv2.imread('Michelangelo/Michelangelo/frag_eroded/frag_eroded_56.png', cv2.IMREAD_UNCHANGED)
img2 = cv2.imread('Michelangelo/Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg', cv2.IMREAD_UNCHANGED)

# Initialiser le détecteur FAST
fast = cv2.FastFeatureDetector_create()

# Initialiser le descripteur BRISK
brisk = cv2.BRISK_create()

# Trouver les points d'intérêt avec FAST
kp1 = fast.detect(img1, None)
kp2 = fast.detect(img2, None)

# Calculer les descripteurs avec BRISK
kp1, des1 = brisk.compute(img1, kp1)
kp2, des2 = brisk.compute(img2, kp2)

# Créer un objet de correspondance de BFMatcher (Brute-Force Matcher)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Trouver les correspondances entre les descripteurs des deux images
matches = bf.match(des1, des2)

# Trier les correspondances en fonction de leur distance
good = [ m for m in matches if m.distance < 50]

# Dessiner les correspondances sur une nouvelle image
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Afficher l'image résultante
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()