import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian

image = cv2.imread('../data/train/27.jpg', 0)

old_image = np.copy(image)
# mild smoothing of the image to reduce noise 
gaussian(image, sigma=0.4)

sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)

# diff the image
image_diff = np.round(sobelx**2 + sobely**2)

print(image_diff)
plt.subplot(2,3,1),plt.imshow(old_image,cmap = 'gray')
plt.title('original image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(image,cmap = 'gray')
plt.title('gaussian image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(image_diff,cmap = 'gray')
plt.title('Differentiated image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
