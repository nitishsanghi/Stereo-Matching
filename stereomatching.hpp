#ifndef STEREOMATCHING_HPP
#define STEREOMATCHING_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// A class designed to perform stereo block matching for computing disparity maps from stereo image pairs.
// This technique is fundamental in computer vision for estimating the depth information of a scene from two images taken from slightly different viewpoints.
class StereoBlockMatching {
private:
    // Unused enum for specifying the disparity type; reserved for future implementations.
    enum DisparityType {
        LEFT,
        RIGHT,
    };

public:
    // Enumeration of different metrics that can be used to calculate disparity between image blocks.
    enum DisparityMetric {
        SAD,  // Sum of Absolute Differences
        ZSAD, // Zero-mean Sum of Absolute Differences
        SSD,  // Sum of Squared Differences
        NCC,  // Normalized Cross-Correlation
        ZNCC, // Zero-mean Normalized Cross-Correlation
    };

    // Default constructor and destructor are defined for proper resource management.
    StereoBlockMatching() = default;
    ~StereoBlockMatching() = default;

    // Sets the parameters required for the disparity calculation process.
    // @param num_disparities: The maximum disparity (the maximum distance between the same physical point in two images).
    // @param block_size: The size of the block used for matching points in two images.
    // @param image_size: The size of the images being processed.
    void setParameters(int num_disparities, int block_size, cv::Size image_size);

    // Computes the disparity map given two stereo images using the specified metric.
    // @param img1: The reference image.
    // @param img2: The target image to compare with the reference image.
    // @param disp: The output disparity map.
    // @param metric: The metric used to calculate the disparity.
    void computeDisparityMap(cv::Mat &img1, cv::Mat &img2, cv::Mat &disp, DisparityMetric metric = SSD);

    // Calculates the number of sliding windows that can be used based on the image column and other parameters.
    // This is used to adjust the computation based on the position within the image to avoid boundary issues.
    // @param n_cols: The total number of columns in the image.
    // @param col: The current column index being processed.
    // @return The number of valid sliding windows for the given column.
    int num_sliding_windows(int n_cols, int col);

    // Calculates the mean value of a given window (block) of the image. Used for ZSAD and ZNCC metrics.
    // @param window: The image block for which to calculate the mean.
    // @return A matrix filled with the mean value, the same size as the input window.
    cv::Mat mean(cv::Mat &window);

    // Calculates the Sum of Squared Differences between two windows.
    // @param window: The first image window.
    // @param window_mean: The mean of the first image window, used when calculating SSD.
    // @param diff: The output matrix that will contain the difference between the two windows.
    // @return The SSD value as a float.
    float ssd(cv::Mat &window, cv::Mat &window_mean, cv::Mat &diff);

    // The following methods calculate disparity using different metrics between two windows.
    float disparity_sad(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff);
    float disparity_zsad(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff);
    float disparity_ssd(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff);
    float disparity_ncc(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff);
    float disparity_zncc(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff);

private:
    // Parameters for the block matching process
    int block_size; // The size of the square block used in matching.
    int num_disparities; // The maximum number of disparity levels to check.
    int padding; // Padding around the image to handle boundary conditions.

    // Caching variables to optimize the disparity calculation by reusing computations.
    float min_ssd; // Minimum SSD value for the current block match.
    int num_windows; // Number of sliding windows for the current column.

    std::vector<int> sliding_windows; // Precomputed sizes of sliding windows for each column.
    cv::Mat disparity_image; // The resulting disparity map.
    
    // Temporary matrices for holding current blocks and calculations.
    cv::Mat ref_window; // Current reference window.
    cv::Mat target_window; // Current target window for comparison.
    cv::Mat ref_window_mean; // Mean of the reference window, used in ZSAD and ZNCC.
    cv::Mat target_window_mean; // Mean of the target window, used in ZNCC.
    
    float ref_ssd; // Sum of squared differences for the reference window, used in NCC and ZNCC.
    cv::Mat ref_ssd_mat; // Matrix form of ref_ssd for calculations.
    
    cv::Mat target_ssd_mat; // SSD values for the target window, used in NCC and ZNCC.
    
    float cross_cor; // Cross-correlation value, used in NCC and ZNCC calculations.
    cv::Mat cross_cor_mat; // Matrix form of cross_cor for calculations.

    const float MIN_SSD_CONSTANT{std::numeric_limits<float>::max()}; // Constant to initialize min_ssd with the maximum float value.
};

#endif // STEREOMATCHING_HPP
