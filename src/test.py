import parser
import segmentation
import cv2
img = parser.get_image('../data/leafscan_selection/', '100142.jpg')
#img[0][0] = [0, 0, 0]
cv2.imshow('img', img)
#print(img)