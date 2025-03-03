# The os module provides functions for interacting with the operating system. The os and os.path module has many functions to interact with files.
import os
import cv2 

# Input and Output Directories
INPUT_DIR = './r'  # Directory with input image 
OUTPUT_DIR = './r_rotated'   # Directory to save rotated and resized image 
ROTATION_ANGLE = 270 # Rotation angles in degree (90, 180, 270)
RESIZE_DIMENSION = (640, 480)   # Width x height for resizing (640 x 480)

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def rotate_and_resize_image(input_path, angle, dimension): 
    """Rotate and resize image
    """
    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not read this image: {input_path}")
        return None
    
    # Rotating the image 
    if angle==90:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle==180:
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    elif angle==270:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Custom angle rotation
        # image.shape -> gives no_of_rows, no_of_columns, channels(only for colored image)
        # ex: image.shape = [100, 120, 3]
        # image[100:120, 240:270] -> rows and column range 
        (h, w) = img.shape[:2]

        # h//2 -> integer value, h/2 -> float value
        center = (w//2, h//2)

        # cv2.getRotationMatrix2d( ) convert image into a 2d matrix 
        # Syntax : getRotationMatrix2d(center, angle, scale)    Angle-> +ve for counterclockwise, -ve for clockwise
        # scale is the scaling factor which scale the image 
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        # cv2.warpAffine() is used to covert matrix to image format
        rotated_img = cv2.warpAffine(img, M, (w, h))

    # Resize the rotated image 
    # syntax -> cv2.resize(source, dimension_size, destination, fx, fy, interpolation)  
    # dsize,dest(array),fx,fy,interpolation are optional 
    # cv2.INTER_AREA shrinks the image 
    resized = cv2.resize(rotated_img, dimension, interpolation=cv2.INTER_AREA)
    return resized





def process_images(input_dir, output_dir, angle, dimension):
    """ Rotate and resize all the images in the input directory and save them to the output directory.
    """
    # os.listdir() is used to get all the files and directory in the specified directory. 
    # If directory is not specified then the list of files and directries in the current working directory will be return.
    files = os.listdir(input_dir)

    for file in files:
        # os.path.join() concatenate various parts of a file path in a way that is independent from the OS.
        # os.path.join(directory or folder_name, variable name of additional path)
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)

        # Rotate and resize the image 
        processed = rotate_and_resize_image(input_path, ROTATION_ANGLE, RESIZE_DIMENSION)

        if processed is not None:
            # Save the processed image 
            cv2.imwrite(output_path, processed)
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Skipping this file: {input_path}") 
        


if __name__ == '__main__':
    process_images(INPUT_DIR, OUTPUT_DIR, ROTATION_ANGLE, RESIZE_DIMENSION)

