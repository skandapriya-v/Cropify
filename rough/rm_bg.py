import os
import cv2
import numpy as np

test_image_path = "leaf2.jpg"

main_img = cv2.imread(test_image_path)
img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)

# Resizing the image
resized_image = cv2.resize(img, (1600, 1200))

# Convert to grayscale
gs = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

# Smoothing the image using Gaussian filter
blur = cv2.GaussianBlur(gs, (55, 55), 0)

# Adaptive thresholding
_, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Closing the holes using morphological Transformation
kernel = np.ones((50, 50), np.uint8)
# dilated = cv2.dilate(im_bw_otsu, kernel, iterations=1)
# eroded = cv2.erode(dilated, kernel, iterations=1)

closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

# Finding contours
contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Finding the largest contour (assuming it's the leaf contour)
cnt = max(contours, key=cv2.contourArea)

# Creating a mask image using the leaf contour
mask = np.zeros_like(resized_image)
cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)

# Masking operation on the original image
maskedImg = cv2.bitwise_and(resized_image, mask)

# Save the background-removed leaf image
os.makedirs("background", exist_ok=True)
img_path = "back_rm.jpg"
cv2.imwrite(f"background/{img_path}", maskedImg)
