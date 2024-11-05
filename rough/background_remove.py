import os
import cv2
import numpy as np
import pandas as pd

test_image_path = "leaf.jpg"

main_img = cv2.imread(test_image_path)

img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)

# resizing the image
resized_image = cv2.resize(img, (1600, 1200))

y, x, _ = img.shape

# convert to gray scale
gs = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

# smoothing the image using gaussian filter

blur = cv2.GaussianBlur(gs, (55, 55), 0)


# adaptive thresholding
ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# closing the holes using morphological Transformation 
kernel = np.ones((50,50),np.uint8)
closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)


# finding contours
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# finding the correct leaf contour
def find_contour(cnts):
    contains = []
    y_ri, x_ri, _ = resized_image.shape
    for cc in cnts:
        yn = cv2.pointPolygonTest(cc, (x_ri // 2, y_ri // 2), False)
        contains.append(yn)

    val = [contains.index(temp) for temp in contains if temp > 0]
    print(contains)
    return val[0]



# creating mask image for background removing using leaf contour
black_img = np.empty([1200,1600,3],dtype=np.uint8)
black_img.fill(0)

index = find_contour(contours)
cnt = contours[index]
mask = cv2.drawContours(black_img, [cnt] , 0, (255,255,255), -1)

# Masking operation on the original image
maskedImg = cv2.bitwise_and(resized_image, mask)

white_pix = [255,255,255]
black_pix = [0,0,0]

final_img = maskedImg
h,w,channels = final_img.shape
for x in range(0,w):
    for y in range(0,h):
        channels_xy = final_img[y,x]
        if all(channels_xy == black_pix):    
            final_img[y,x] = white_pix

# now save the background removed leaf image
os.makedirs("background",exist_ok=True)
img_path = "back_rm.jpg"
cv2.imwrite(f"background/{img_path}",final_img)

