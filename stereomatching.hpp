#ifndef STEREOMATCHING_HPP
#define STEREOMATCHING_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>


// Define a class for performing stereo block matching, which is used for computing disparity maps from stereo images.
class StereoBlockMatching{
    private:
        // Enum to specify the disparity type, though it's not used in the current implementation.
        enum DisparityType {
            LEFT,
            RIGHT,
        };

        enum DisparityMetric {
            SAD,
            SSD,
            NCC,
            ZNCC,
        };

    public:
        // Default constructor and destructor. They do nothing but are good practice for resource management.
        StereoBlockMatching() = default;
        ~StereoBlockMatching() = default;

        // Set parameters for the stereo block matching process including the number of disparities, block size, and the image size.
        void setParameters(int num_disparities, int block_size, cv::Size image_size);

        // Compute the disparity map given two stereo images.
        void computeDisparityMap(cv::Mat &img1, cv::Mat &img2, cv::Mat &disp, DisparityMetric metric = SSD);

        // Calculate the number of sliding windows based on the image column and other parameters.
        int num_sliding_windows(int n_cols, int col);

        float disparity_ssd(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff);

        cv::Mat mean(cv::Mat &window);

        float ssd(cv::Mat &window, cv::Mat &window_mean, cv::Mat &diff);

        float disparity_sad(cv::Mat &window, cv::Mat &window_mean, cv::Mat &diff);
        
    private:
        // Private member variables to store the block size, number of disparities, padding, the disparity image, and the sliding windows calculations.
        int block_size;
        int num_disparities;
        int padding;
        cv::Mat disparity_image;
        std::vector<int> sliding_windows;        
};

#endif // STEREOMATCHING_HPP