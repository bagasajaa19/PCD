import imageio.v3 as image
import numpy as np
import matplotlib.pyplot as plt

# =========================
# BACA GAMBAR
# =========================
path = "motor.jpeg"
my_image = image.imread(path)

# =========================
# FUNGSI BRIGHTNESS
# =========================
def brightness(img, factor):
    img_result = img.astype(np.float32)
    img_result = img_result + factor
    img_result = np.clip(img_result, 0, 255)
    return img_result.astype(np.uint8)

# =========================
# FUNGSI CONTRAST
# =========================
def contrast(img, factor):
    mean = 90

    img_result = img.astype(np.float32)

    img_result = mean + factor * (img_result - mean)

    img_result = np.clip(img_result, 0, 255)

    return img_result.astype(np.uint8)

# =========================
# PROSES CITRA
# =========================
bright = brightness(my_image, -160)
contrast_img = contrast(my_image, 28)

# =========================
# TAMPILKAN 3 GAMBAR
# =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

axes[0].imshow(my_image)
axes[0].set_title("Gambar Asli")

axes[1].imshow(bright)
axes[1].set_title("Brightness -160")

axes[2].imshow(contrast_img)
axes[2].set_title("Contrast 28")

for ax in axes:
    ax.axis("on")

plt.tight_layout()

# =========================
# TAMPILKAN RGB SAAT CURSOR BERGERAK
# =========================
def format_coord(x, y):

    x = int(x)
    y = int(y)

    if (
        x >= 0 and
        y >= 0 and
        x < my_image.shape[1] and
        y < my_image.shape[0]
    ):
        r, g, b = my_image[y, x]

        return (
            f"X={x} Y={y} | "
            f"R={r} G={g} B={b}"
        )

    return f"X={x} Y={y}"

axes[0].format_coord = format_coord

# =========================
# EVENT KLIK PIXEL
# =========================
def onclick(event):

    if event.inaxes == axes[0]:

        x = int(event.xdata)
        y = int(event.ydata)

        if (
            x >= 0 and
            y >= 0 and
            x < my_image.shape[1] and
            y < my_image.shape[0]
        ):

            r, g, b = my_image[y, x]

            print(
                f"Koordinat ({x},{y}) "
                f"= R:{r} G:{g} B:{b}"
            )

fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()

