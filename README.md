# Panoramic-Image-Creation-Project

Creating a panoramic image with a combination of multiple images.

This repository contains scripts to create a full panoramic image by combining multiple images. The process involves counting the number of images, rotating them if necessary, and then stitching them together to form a seamless panorama.

**Workflow**

*Count Images (no_of_image.cpp):*
This script counts the total number of images in a specified folder.

*Rotate Images (rotate_image.py):*
If any images need to be rotated to maintain the correct left-to-right order, this script adjusts their orientation accordingly.

*Create Panorama (panorama_image.py):*
Combines the processed images into a final panoramic image.

**Requirements**

*Make sure you have the following dependencies installed:*
OpenCV (cv2)

NumPy

Imutils

os

file

#include <vector>

#include <filesystem>   

#include <opencv2/opencv.hpp>

C++ compiler (for no_of_image.cpp)

**Usage**

*Compile and run no_of_image.cpp to count the images:*

g++ no_of_image.cpp -o count_images

./count_images

*Run rotate_image.py to correct the orientation:*

python3 rotate_image.py

*Run panorama_image.py to generate the panoramic image:*

python3 panorama_image.py

**Output**

The final panoramic image will be saved in the output folder.
