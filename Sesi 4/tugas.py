import imageio.v2 as imageio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Membaca citra
img = imageio.imread('motor.jpeg')

# Konversi ke grayscale jika RGB
if len(img.shape) == 3:
    img_gray = np.mean(img, axis=2).astype(np.uint8)
else:
    img_gray = img

# Filtering menggunakan Gaussian Filter
img_blur = ndimage.gaussian_filter(img_gray, sigma=2)

# Histogram Equalization
hist, bins = np.histogram(img_blur.flatten(), 256, [0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * 255 / cdf[-1]

img_equalized = np.interp(
    img_blur.flatten(),
    bins[:-1],
    cdf_normalized
).reshape(img_blur.shape)

# Menampilkan hasil
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.imshow(img_gray, cmap='gray')
plt.title('Citra Awal')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(img_blur, cmap='gray')
plt.title('Gaussian Filter')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(img_equalized, cmap='gray')
plt.title('Histogram Equalization')
plt.axis('off')

plt.tight_layout()
plt.show()