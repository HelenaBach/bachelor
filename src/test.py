import parser
import segmentation
import cv2
img = parser.get_image('../../Data/leg/', '100142.jpg')
img[0][0] = [0, 0, 0]
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(img)
