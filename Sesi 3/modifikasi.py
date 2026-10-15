import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

nama_file = "foto.jpeg"

# 1. Membaca citra
img = imageio.imread(nama_file)
print("Citra RGB    :", img.shape)

# 2. Konversi RGB → Grayscale
R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
img_gray = (0.299*R + 0.587*G + 0.114*B).astype(np.uint8)
print("Citra Grayscale  :", img_gray.shape)

# 3. Hitung piksel
tinggi, lebar = img_gray.shape
jumlah_piksel = img_gray.size
print("Tinggi :", tinggi, "px")
print("Lebar  :", lebar, "px")
print("Jumlah :", jumlah_piksel, "px")

# 4. Histogram intensitas
histogram = np.bincount(img_gray.flatten(), minlength=256)

# 5. Intensitas dominan
intensitas_dominan = np.argmax(histogram)
jumlah_intensitas = histogram[intensitas_dominan]
print("Intensitas Dominan :", intensitas_dominan)
print("Jumlah Intensitas  :", jumlah_intensitas)

# 6. Citra negatif
img_negatif = 255 - img_gray

# 7. Visualisasi
plt.figure(figsize=(16,6))

# Citra asli
plt.subplot(1,3,1)
plt.imshow(img)
plt.title("Citra Asli")

# Citra grayscale
plt.subplot(1,3,2)
plt.imshow(img_gray, cmap="gray")
plt.title(f"Grayscale ({tinggi}x{lebar}px)")

# Histogram
plt.subplot(1,3,3)
plt.bar(range(256), histogram, width=1, color="skyblue")
plt.axvline(intensitas_dominan, color="red", linestyle="--", label=f"Dominan: {intensitas_dominan}")
plt.legend()
plt.ylabel("Jumlah Piksel")
plt.xlabel("Nilai Intensitas")
plt.title("Histogram Intensitas")

plt.tight_layout()
plt.show()

# 8. Tampilkan citra negatif
plt.figure(figsize=(6,6))
plt.imshow(img_negatif, cmap="gray")
plt.title("Citra Negatif")
plt.axis("off")
plt.show()