#include <iostream>
#include <vector>
// Performing files and directory operation was a bulky and mistake susceptible task as it required the use of platform specific function and libraries.
// The filesystem library was added to copy these troubles, offering a portable and standarized way to paint with the file system.
#include <filesystem> 
#include <opencv2/opencv.hpp>

std::vector<cv::Mat> paroram_img;

int read_img(std::string directory){
    // Initial number of count of images = 0 
    int count_img = 0;       

    // If the directory of images file exists then count number of images else return number of images = 0
    if (std::filesystem::exists(directory) && std::filesystem::is_directory(directory)){

        // Iterating in the directory of images file 
        for (const auto& entry : std::filesystem::directory_iterator(directory)){
            // Path of each image file in the directory 
            std::string filePath = entry.path().string();
            // Reading the image file and storing it into the vector
            cv::Mat img = cv::imread(filePath);
            if (img.empty()){
                std::cerr << "Failed to read image : " << filePath << std::endl;
            }
            else{
                count_img++;
                paroram_img.push_back(img);
                std::cout << "Loaded Image :" << filePath << std::endl;
            }
        }
    }
    return count_img;

}

int main(){
    std::string directory = "./c2/";
    int no_of_image = read_img(directory);
    std::cout << "No of Images : " << no_of_image << std::endl;
}
