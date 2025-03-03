import numpy as np
import cv2
import imutils 
import time 
import os


# DIR = './RotatedImages'
# DIR = './ProcessedImages'
DIRECTORY = './r_rotated'

R_WIDTH = 4000
WIDTH = 8002
HEIGHT = 4001
BLACK_COLOR = 25    # We will remove the pixels column and rows which have black color box with value>25
RESULT = './combine.jpg' 

#os.makedirs(DIRECTORY, exist_of=True)

files = os.listdir(DIRECTORY)


def complement_sky(image):
    tmp = imutils.resize(image, width=WIDTH)
    row, col = tmp.shape[:2]
    border = HEIGHT - row

    sky = cv2.imread('sky.jpg')
    sky = imutils.resize(sky, width=WIDTH)
    sky_rows = sky.shape[0]
    start = sky_rows - border
    sky = sky[start:sky_rows]

    # Stacking arrays in a single array using np.vstack()
    img = np.vstack([sky[:,:], tmp[:,:]])    #np.vstack(sky[:,:], tmp[:,:])

    mark = np.zeros((HEIGHT, WIDTH), np.uint8)
    color = (0, 0, 0)
    mark[0:border, 0:col] = 255

    img = cv2.inpaint(img, mark, 3, cv2.INPAINT_TELEA)

    tmp_sky = img[0:border, :]
    sky = cv2.addWeighted(tmp_sky, 0.7, sky, 0.3, 0.0)
    img = np.vstack([sky[:,:], img[border:,:]])      #np.vstack(sky[:,:], img[border:,:])

    start = border-1
    end = start-100
    blend = 0.01
    
    for r in range(start, end, -1):
        img[r,:] = tmp[0,:] * (1-blend) + sky[r,:] * blend
        blend = blend + 0.01

    rows, cols = image.shape[:2]
    img = img[1:, 1:cols-1]

    tmp = img[0:border+100, :]
    tmp = cv2.GaussianBlur(tmp, (9,9), 2.5)
    img = np.vstack([tmp[:,:], img[border+100:,:]])      #np.vstack(tmp[:,:], img[border+100:,:])

    return img
 


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
            if sum(image[row, col]) < BLACK_COLOR:
                row = row+1
            else:
                if row > top:
                    top = row
                break
        col = col+1
    top = top+1
 
    
    # Bottom Crop
    col = 0
    while col < width:
        row = height-1
        while row > bottom_limit:
            if sum(image[row, col]) < BLACK_COLOR:
                row = row-1
            else:
                if bottom > row:
                    bottom = row
                break;
        col = col+1
    bottom = bottom-1


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
            if sum(image[row, col]) < BLACK_COLOR:
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
            if sum(image[row, col]) < BLACK_COLOR:
                col = col-1
            else:
                if col < right:
                    right = col
                break
        row = row+1
    
    image = image[0:height, left:right]
    return image



def stitch(files):
    images = []
    for file in files:
        img = cv2.imread(DIRECTORY + '/' + file)
        if img is None:
            print(f"Error loading image: {file}")
        else:
            images.append(img)
        #images.append(cv2.imread(DIRECTORY + '/' + file))
    # try_use_gpu is used in Stitcher class to control whether to use GPU acceleration for image stitching.
    try_use_gpu = False

    #for i in range(len(images)):
    #     images[i] = cv2.resize(images[i], (480, 640))

    #for i in range(len(images)):
    #    images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)

    stitcher = cv2.Stitcher_create()    #cv2.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(images)
    if status==0:
        return panorama
    else:
        print(status)
        return None




if __name__ == '__main__':
    # time.time() gives the current time in seconds from the epoch, where epoch is 1 jan 1970 00:00:00
    # It gives a floating value in seconds(high precision value)
    start = time.time()
    files = os.listdir(DIRECTORY)
    panorama = stitch(files)
    if panorama is not None:
        panorama = crop(panorama)
        #panorama = complement_sky(panorama)
        cv2.imwrite(RESULT, panorama)
    else:
        print('error')
    end = time.time()
    print("Cost " + str(end-start))
