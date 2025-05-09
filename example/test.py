import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.segmentation import chan_vese

# === Načti a připrav obrázek ===
img_bgr = cv2.imread("images/B/23.JPG")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rgb_float = img_as_float(img_rgb)  # rozsah 0–1

# === Chan-Vese segmentace na každý kanál ===
print("Segmentuji R kanál...")
cv_r = chan_vese(
    img_rgb_float[:, :, 0],
    mu=0.25, lambda1=1, lambda2=1,
    tol=1e-3, max_num_iter=200, dt=0.5,
    init_level_set="checkerboard", extended_output=False
)

print("Segmentuji G kanál...")
cv_g = chan_vese(
    img_rgb_float[:, :, 1],
    mu=0.25, lambda1=1, lambda2=1,
    tol=1e-3, max_num_iter=200, dt=0.5,
    init_level_set="checkerboard", extended_output=False
)

print("Segmentuji B kanál...")
cv_b = chan_vese(
    img_rgb_float[:, :, 2],
    mu=0.25, lambda1=1, lambda2=1,
    tol=1e-3, max_num_iter=200, dt=0.5,
    init_level_set="checkerboard", extended_output=False
)

# === Kombinuj výsledky – alespoň 2 z 3 kanálů musí souhlasit ===
cv_sum = cv_r.astype(int) + cv_g.astype(int) + cv_b.astype(int)
cv_combined = cv_sum >= 2

# === Vizuální výstup ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Původní obrázek")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Spojená segmentace (RGB)")
plt.imshow(cv_combined, cmap='gray')
plt.axis('off')

# Maskuj výsledek na originál
masked = img_rgb.copy()
masked[~cv_combined] = 0

plt.subplot(1, 3, 3)
plt.title("Vyříznuté graffiti")
plt.imshow(masked)
plt.axis('off')

plt.tight_layout()
plt.show()
