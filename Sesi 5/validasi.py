import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("motor.jpeg")
if img is None:
    raise FileNotFoundError("motor.jpeg tidak ditemukan")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mean = cv2.blur(rgb,(3, 3))
noisse = cv2.add(rgb, cv2.randn(np.zeros_like(rgb), 15, 50))
gauss = cv2.GaussianBlur(rgb,(3, 3), 0)
median = cv2.medianBlur(rgb, 3)

fig, ax = plt.subplots(1, 5, figsize=(18, 4))
for a, im, t in zip(ax, [rgb, mean, noisse, gauss, median],
                    ["Asli", "Mean", "Noise", "Gaussian", "Median"]):
    a.imshow(im); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()