import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import models, transforms

# === Načti obrázek ===
img = cv2.imread("images/13.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === Předzpracování obrázku pro model ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

img_tensor = transform(img_rgb)
img_tensor = img_tensor.unsqueeze(0)  # Přidejte batch dimenzi

# === Načti předtrénovaný model Mask R-CNN ===
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Nastavení modelu do eval režimu

# === Inference ===
with torch.no_grad():
    prediction = model(img_tensor)

# === Zpracování výsledků ===
# prediction[0] obsahuje výsledky pro obrázek
masks = prediction[0]['masks'] > 0.5  # Filtruj masky, které mají vyšší prahovou hodnotu než 0.5
labels = prediction[0]['labels']
scores = prediction[0]['scores']

# === Zobraz výsledek ===
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(img_rgb)

# Zobrazíme masky pro každý objekt
for mask, label, score in zip(masks, labels, scores):
    mask = mask.squeeze().cpu().numpy()  # Naformátuj masku pro zobrazení
    if score > 0.5:  # Filtruj objekty podle skóre
        color = np.random.rand(3,)  # Generuj náhodnou barvu pro každou masku
        ax.imshow(mask, alpha=0.5, cmap='jet')  # Zobrazíme masku s průhledností

ax.axis('off')
plt.title("Detekované objekty (Mask R-CNN z torchvision)")
plt.show()
