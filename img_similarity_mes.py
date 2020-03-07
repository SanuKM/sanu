
# 1. Import the necessary packages
from skimage.measure import compare_ssim
from PIL import Image 
from resizeimage import resizeimage
import argparse
import imutils
import cv2
import numpy as np

# get the image the two images 
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="first")
ap.add_argument("-s", "--second", required=True, help="second")
args = vars(ap.parse_args())

#read and save the these images
image1 = cv2.imread(args["first"])
cv2.imwrite('imgA1.png',image1)
image2 = cv2.imread(args["second"])
cv2.imwrite('imgA2.png',image2)

#resize the both image
with open('imgA1.png', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [400, 500])
        cover.save('1st_image.png', image.format)
imgAA1 = cv2.imread('1st_image.png')

with open('imgA2.png', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [400, 500])
        cover.save('2nd_image.png', image.format)
imgAA2 = cv2.imread('2nd_image.png')


#convert the image into gray
img1 = cv2.imread('1st_image.png',0)
img2 = cv2.imread('2nd_image.png',0)

#Apply Median Filtering for removing salt-and-pepper noise
med1 = cv2.medianBlur(img1,5)
med2 = cv2.medianBlur(img2,5)

# Structural Similarity Index - for find out the Similarity of two image
(score, diff) = compare_ssim(med1, med2, full=True)
diff = (diff * 255).astype("uint8")

# Print the Score 
image_Similarity = np.zeros((512,512,3),np.uint8)

myimg = cv2.putText(image_Similarity,"For Total Similarity, SSIM =1", (20,90),cv2.FONT_ITALIC,1,(220,130,250),2)
myimg = cv2.putText(image_Similarity,"For total dissimilarity, SSIM =0", (40,50),cv2.FONT_ITALIC,1,(220,130,250),2)
myimg = cv2.putText(image_Similarity,"SSIM:{}".format(score), (10,290),cv2.FONT_ITALIC,2,(220,130,250),2)

print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to obtain the regions of the two input images that differ
thresh = cv2.threshold(diff,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour and then draw the bounding box on both input images to represent where the two images differ
    (x,y,w,h) = cv2.boundingRect(c) 
    cv2.rectangle(imgAA1, (x,y), (x+w,y+h),(0,0,225),2)
    cv2.rectangle(imgAA2, (x,y), (x+w,y+h),(0,0,255),2)

# show the output images
cv2.imshow("1st image",imgAA1)
cv2.imshow("2nd image",imgAA2)
cv2.imshow("image_Similarity",myimg)

cv2.waitKey()
cv2.destroyAllWindows()