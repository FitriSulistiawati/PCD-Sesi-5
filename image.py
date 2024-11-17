import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menampilkan citra
def display_images(images, titles, cmap=None):
    plt.figure(figsize=(15, 8))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        if cmap == 'gray':
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Membaca citra input
image_path = "C:\\Users\\Lenovo\\Pictures\\2516737fddee12a2379181dd90a0725c.jpg"  
image_color = cv2.imread(image_path)
if image_color is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan di jalur: {image_path}")
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# 1. Filter Low-Pass (Gaussian Blur)
low_pass_color = cv2.GaussianBlur(image_color, (11, 11), 0)
low_pass_gray = cv2.GaussianBlur(image_gray, (11, 11), 0)

# 2. Filter High-Pass (Laplacian)
high_pass_color = cv2.Laplacian(image_color, cv2.CV_64F)
high_pass_gray = cv2.Laplacian(image_gray, cv2.CV_64F)

# Konversi ke 8-bit untuk visualisasi
high_pass_color = cv2.convertScaleAbs(high_pass_color)
high_pass_gray = cv2.convertScaleAbs(high_pass_gray)

# 3. Filter High-Boost
alpha = 1.5  # Parameter penguatan
high_boost_color = cv2.addWeighted(image_color, alpha, low_pass_color, 1 - alpha, 0)
high_boost_gray = cv2.addWeighted(image_gray, alpha, low_pass_gray, 1 - alpha, 0)

# Menampilkan hasil untuk citra berwarna
display_images(
    [image_color, low_pass_color, high_pass_color, high_boost_color],
    ["Original (Color)", "Low-Pass (Color)", "High-Pass (Color)", "High-Boost (Color)"]
)

# Menampilkan hasil untuk citra grayscale
display_images(
    [image_gray, low_pass_gray, high_pass_gray, high_boost_gray],
    ["Original (Grayscale)", "Low-Pass (Grayscale)", "High-Pass (Grayscale)", "High-Boost (Grayscale)"],
    cmap='gray'
)
