import numpy as np
# Importing OpenCV(cv2) module 
import cv2
import glob
import imutils 
import time


R_WIDTH = 4000
WIDTH = 8002
HEIGHT = 4001
BLACK_COLOR = 50


def crop(image):
    # (height, width) = image.shape[:2]
    height, width = image.shape[0], image.shape[1]

    if width > R_WIDTH:
        # imutils.resize(image, w) only be able to specify either height or width
        # but cv2.resize(image, (w, h)) can resize both height and width 
        image = imutils.resize(image, width=R_WIDTH)
        height, width = image.shape[0], image.shape[1]

    top, bottom = 0, height
    limit = height//8   # int(height/8)
    top_limit = limit
    bottom_limit = height-limit 

    # Top Crop
    col = 0
    while col < width:
        row = 0
        while row < top_limit:
            if np.sum(image[row, col]) < BLACK_COLOR:
                row = row+1
            else:
                if row > top:
                    top = row
                break
        col = col+1
    #top = top+1
 
    
    # Bottom Crop
    col = 0
    while col < width:
        row = height-1
        while row > bottom_limit:
            if np.sum(image[row, col]) < BLACK_COLOR:
                row = row-1
            else:
                if bottom > row:
                    bottom = row
                break
        col = col+1
    #bottom = bottom-1


    image = image[top:bottom, 0:width]
    (height, width) = image.shape[:2]
    left, right = 0, width
    limit = int(width/8)
    left_limit = limit
    right_limit = width - limit 


    # Left Crop 
    row = 0
    while row < height:
        col = 0
        while col < left_limit:
            if np.sum(image[row, col]) < BLACK_COLOR:
                col = col+1
            else:
                if col > left:
                    left = col
                break
        row = row+1
 

    # Right Crop 
    row = 0
    while row < height:
        col = width-1
        while col > right_limit:
            if np.sum(image[row, col]) < BLACK_COLOR:
                col = col-1
            else:
                if col < right:
                    right = col
                break
        row = row+1
    
    image = image[0:height, left:right]
    return image




image_paths = ['r1.jpg', 'r2.jpg']
images = []

# Path of all the images
# print(image_paths)       
for image in image_paths:

    # Reading the image
    # Syntax -> cv2.imread(source, flag)     flag->How image should be read
    img = cv2.imread(image)
    images.append(img)

    # Output img with window name image 
    # Syntax -> cv2.imshow(window_name, image)
    #cv2.imshow('image', img)

    # Wait for the user to press a key (Maintain output window until user presses a key)
    #cv2.waitKey(0)


# A function that create a stitcher object that can combine multiple images into panorama
imageStitcher = cv2.Stitcher_create()

# If we are not able to stitch image together (when it couldn't find any key point between images)
error, stitched_img = imageStitcher.stitch(images)

if not error:
    cv2.imwrite("StitchedOutput.png", stitched_img)
    #cv2.imshow("Stitched Image", stitched_img)
    #cv2.waitKey(0)

    # cv2.copyMakeBorder used to create a border around the image like a photo frame 
    # Adjusting the border of the stitched image (10 pixels at each corner)
    # cv2.BORDER_CONSTANT adds a constant colored border
    # Syntax -> cv2.copyMakeBorder(source, top, bottom, left, right, borderType, value)
    # top, bottom, left, right are in pixels
    #stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
    
    # Converting RGB to Gray scale for fixing threshold value
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)

    # Treshold function of the pixel- classifying the image into distinct regions based on pixel intensity, (it is done on gray scale images)
    # If pixelValue < ThresholdValue the set to 0, otherwise set max 255 (usually) for cv2.THRESH_BINARY.
    # Pixcel -> Pixcel(picture, element) is tiny dot or square that maked up a digital image.
    # Syntax -> cv2.thresholf(source, thresholdValue, MaxValue, thresholdTechnique)
    thresh_img = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1] # 0 (black)

    #cv2.imshow('Threshold Image', thresh_img)
    #cv2.waitKey(0)

    # Contours is a line joining all the points along the boundary of an image that are having same intensity (works best on binary images)
    # Using thresh_img.copy() because findContours alters the image 
    # This will stored in list, each list will have different contours
    # Syntax -> cv2.findContours(binary_image, cv2.RETR_EXTERNAL (retrival mode) or cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE (approximation mode))
    #contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    contours = imutils.grab_contours(contours)
    # Getting the maximum area of object in pixel,
    # One contour with the stitched image together will be the maximum area of the image 
    areaOI = max(contours, key=cv2.contourArea)     # AreaOfInterest will be the actual image that we stitched together

    # unit8 is unsigned integer of 8 bit with a range of 0 to 255 (because we are working with binary image)
    mask = np.zeros(thresh_img.shape, dtype='uint8')

    # cv2.boundingRect is used to draw an approximate rectangle around the binary image (avoid outside black parts)
    x, y, w, h = cv2.boundingRect(areaOI)

    # Drawing the bounding rectangle (croping our image)
    cv2.rectangle(mask, (x,y), (x+w, y+h), 255, -1)
    #stitched_img = stitched_img[y:y+h, x:x+w]

    # Creating kernal 
    minRectangle = mask.copy()
    sub = mask.copy()       # Subtraction

    while cv2.countNonZero(sub)>0:
        # Erode is like a soil erosion for image 
        #minRectangle = cv2.erode(minRectangle, None)
        minRectangle = cv2.erode(minRectangle, None, iterations=1)  # Reduce iterations

        # Arithmatic operators like adition, subtraction make image brighter or darker 
        # The filters which we take to use of different selfies and photos use image substraction
        # Syntax -> cv2.subtract(image1, image2)    Both images should be of equal size
        sub = cv2.subtract(minRectangle, thresh_img)    # This will find difference between 2 images


    cv2.imwrite("debug_thresh_img.jpg", thresh_img)
    cv2.imwrite("debug_minRectangle.jpg", minRectangle)

    #contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    #areaOI = max(contours, key=cv2.contourArea)
    if contours:  # Check if contours list is not empty
        areaOI = max(contours, key=cv2.contourArea)
    else:
        print("No contours found! Exiting...")
        exit()  # Or handle accordingly


    #cv2.imshow("minRectangle Image", minRectangle)
    #cv2.waitKey(0)

    x, y, w, h = cv2.boundingRect(areaOI)

    # Crop our image to area of interest 
    stitched_img = stitched_img[y:y+h, x:x+w]
    #stitched_img = crop(stitched_img)
    cv2.imwrite("StitchedOutputProcessed12.jpg", stitched_img)
    #cv2.imshow("Stitched Image Processed", stitched_img)

    #cv2.waitKey(0)

else:
    print("Image could not be stitched together!")
    print("Likely not enough keypoint being detected!")

















'''import numpy as np
# Importing OpenCV(cv2) module 
import cv2
import glob
import imutils 

image_paths = ['r1.jpg', 'r2.jpg']
images = []

# Path of all the images
# print(image_paths)       
for image in image_paths:

    # Reading the image
    # Syntax -> cv2.imread(source, flag)     flag->How image should be read
    img = cv2.imread(image)
    images.append(img)

    # Output img with window name image 
    # Syntax -> cv2.imshow(window_name, image)
    #cv2.imshow('image', img)

    # Wait for the user to press a key (Maintain output window until user presses a key)
    #cv2.waitKey(0)


# A function that create a stitcher object that can combine multiple images into panorama
imageStitcher = cv2.Stitcher_create()

# If we are not able to stitch image together (when it couldn't find any key point between images)
error, stitched_img = imageStitcher.stitch(images)

if not error:
    #cv2.imwrite("StitchedOutput.png", stitched_img)
    #cv2.imshow("Stitched Image", stitched_img)
    #cv2.waitKey(0)

    # cv2.copyMakeBorder used to create a border around the image like a photo frame 
    # Adjusting the border of the stitched image (10 pixels at each corner)
    # cv2.BORDER_CONSTANT adds a constant colored border
    # Syntax -> cv2.copyMakeBorder(source, top, bottom, left, right, borderType, value)
    # top, bottom, left, right are in pixels
    #stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
    
    # Converting RGB to Gray scale for fixing threshold value
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)

    # Treshold function of the pixel- classifying the image into distinct regions based on pixel intensity, (it is done on gray scale images)
    # If pixelValue < ThresholdValue the set to 0, otherwise set max 255 (usually) for cv2.THRESH_BINARY.
    # Pixcel -> Pixcel(picture, element) is tiny dot or square that maked up a digital image.
    # Syntax -> cv2.thresholf(source, thresholdValue, MaxValue, thresholdTechnique)
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1] # 0 (black)

    #cv2.imshow('Threshold Image', thresh_img)
    #cv2.waitKey(0)

    # Contours is a line joining all the points along the boundary of an image that are having same intensity (works best on binary images)
    # Using thresh_img.copy() because findContours alters the image 
    # This will stored in list, each list will have different contours
    # Syntax -> cv2.findContours(binary_image, cv2.RETR_EXTERNAL (retrival mode) or cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE (approximation mode))
    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    # Getting the maximum area of object in pixel,
    # One contour with the stitched image together will be the maximum area of the image 
    areaOI = max(contours, key=cv2.contourArea)     # AreaOfInterest will be the actual image that we stitched together

    # unit8 is unsigned integer of 8 bit with a range of 0 to 255 (because we are working with binary image)
    mask = np.zeros(thresh_img.shape, dtype='uint8')

    # cv2.boundingRect is used to draw an approximate rectangle around the binary image (avoid outside black parts)
    x, y, w, h = cv2.boundingRect(areaOI)

    # Drawing the bounding rectangle (croping our image)
    cv2.rectangle(mask, (x,y), (x+w, y+h), 255, -1)
    #stitched_img = stitched_img[y:y+h, x:x+w]

    # Creating kernal 
    minRectangle = mask.copy()
    sub = mask.copy()       # Subtraction

    while cv2.countNonZero(sub)>0:
        # Erode is like a soil erosion for image 
        minRectangle = cv2.erode(minRectangle, None)

        # Arithmatic operators like adition, subtraction make image brighter or darker 
        # The filters which we take to use of different selfies and photos use image substraction
        # Syntax -> cv2.subtract(image1, image2)    Both images should be of equal size
        sub = cv2.subtract(minRectangle, thresh_img)    # This will find difference between 2 images


    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    #cv2.imshow("minRectangle Image", minRectangle)
    #cv2.waitKey(0)

    x, y, w, h = cv2.boundingRect(areaOI)

    # Crop our image to area of interest 
    stitched_img = stitched_img[y:y+h, x:x+w]
    cv2.imwrite("StitchedOutputProcessed.jpg", stitched_img)
    #cv2.imshow("Stitched Image Processed", stitched_img)

    #cv2.waitKey(0)

else:
    print("Image could not be stitched together!")
    print("Likely not enough keypoint being detected!")
'''