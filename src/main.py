import segmentation
import os
import cv2

path = '../data/leafscan/'
images = os.listdir(path)[:100]
img_4_landmarks = []

for image in images:
	if image.endswith('.jpg'):
	    # indlæs billedet i greyscale (det er det 0 betyder)
	    img = cv2.imread('../data/leafscan/' + image,0)
	    # gør Otsu agtige ting
	    ret,thr =cv2.threshold(img,0,255,cv2.THRESH_OTSU) #Ved ikke lige hvad det ægte foregår her
	    img_4_landmarks.append((thr, img))

for thr, img in img_4_landmarks:
	landmarks = segmentation.landmark_setter(thr, img)


