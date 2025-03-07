import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('r1.jpg')          # queryImage
img2 = cv.imread('r2.jpg') # trainImage

# Initiate SIFT detector
# Creates a SIFT (Scale-Invariant Feature Transform) detector, which is used to detect keypoints and compute descriptors in images.
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
# FLANN (Fast Library for Approximate Nearest Neighbors) is used for fast feature matching.
FLANN_INDEX_KDTREE = 1  # Specifies that KD-Tree should be used (suitable for SIFT descriptors).
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)  # Number of trees in KD-Tree.
search_params = dict(checks=50)   # or pass empty dictionary    # Limits the number of checks to 50 to balance speed and accuracy.

flann = cv.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
# Ensures that a match is good by checking if the best match (m.distance) is significantly better than the second-best (n.distance).
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

res = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

#plt.imshow(img3,),plt.show()

plt.imshow(res)
plt.axis('off')  # Hide axis for better visualization
plt.savefig("output_image2.jpg")  # Save the image
plt.close()  # Close the plot to free memory