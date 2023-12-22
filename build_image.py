import cv2 as cv
import os
import numpy as np

path = "Michelangelo/Michelangelo/frag_eroded/"

# Charger les données des fragments
solution=['results.txt','results2.txt']
output=["ransac_output.png","euclidian_output.png"]
i_solution = int(input("Even number to build RANSAC solution, odd number for Euclidian "))
fragment_data = [line.strip().split() for line in open(solution[i_solution%2], "r")]

img_width, img_height = 2000, 1500

# Créer une image vide de la taille de l'image de sortie
image = np.zeros((img_height, img_width, 3), np.uint8)  # Utilisez 3 canaux pour RGB (fond noir, chaque pixel à zéro)

# Placer les fragments dans l'image de sortie avec une marge à partir de (200, 200)
margin_x, margin_y = 200, 200

for i in fragment_data:
    index, center_x, center_y, angle = i
    if (int(center_x),int(center_y),float(angle))==(0,0,0.):
        continue
    fragment_filename = os.path.join(path, f"frag_eroded_{int(index)}.png")

    # Charger les fragments
    fragment = cv.imread(fragment_filename)

    fragment_height, fragment_width, _ = fragment.shape

    M = cv.getRotationMatrix2D(
        (fragment_width / 2, fragment_height / 2),
        float(angle),
        1,
    )
    fragment = cv.warpAffine(fragment, M, (fragment_width, fragment_height))

    # Calculer les coins du fragment avec la marge
    x1 = int(center_x) - fragment_width // 2 + margin_x
    y1 = int(center_y) - fragment_height // 2 + margin_y
    x2 = x1 + fragment_width
    y2 = y1 + fragment_height

    # Ajouter la valeur des pixels si colorés
    try:
        fragment_mask = (fragment != 0)
        image[y1:y2, x1:x2][fragment_mask] = fragment[fragment_mask]
        image = np.clip(image, 0, 255) 
    except:
        print(fragment.shape)

# crop l'image 1775x775
crop_img = image[200:975, 200:1907]

# Afficher l'image de sortie
cv.imwrite(output[i_solution%2], crop_img)
cv.imshow("Image Finale", crop_img)
cv.waitKey(0)
cv.destroyAllWindows()