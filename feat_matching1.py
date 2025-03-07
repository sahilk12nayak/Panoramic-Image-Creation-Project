import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('r1.jpg')          # queryImage
img2 = cv.imread('r2.jpg') # trainImage

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
res = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#plt.imshow(res),plt.show()

plt.imshow(res)
plt.axis('off')  # Hide axis for better visualization
plt.savefig("output_image1.jpg")  # Save the image
plt.close()  # Close the plot to free memory
