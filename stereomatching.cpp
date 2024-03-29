#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <numeric>
#include <chrono>


class StereoBlockMatching{
    public:
        StereoBlockMatching();
        ~StereoBlockMatching();
        void setParameters(int num_disparities, int block_size, cv::Size image_size);
        void makeGrayscale(cv::Mat &img, cv::Mat &gray_img);
        void addPadding(cv::Mat &img, cv::Mat &padded_img);
        void computeDisparityMap(cv::Mat &img1, cv::Mat &img2, cv::Mat &disp);
        int num_sliding_windows(int n_cols, int col, std::string disparity_type);
        std::vector<int> num_sliding_windows_v2(int n_cols, int col);
        void target_window_stacker(cv::Mat &target_padded, int col, int row, std::string disparity_type, int sliding_windows);

    private:
        int block_size;
        int num_disparities;
        int padding;
        cv::Mat disparity_image;
        std::vector<std::vector<int>> sliding_windows;
        std::vector<cv::Mat> ref_window_stack;
        std::vector<cv::Mat> target_window_stack;
        std::vector<int> ssd_values;
        
};

StereoBlockMatching::StereoBlockMatching(){
    // Constructor
}

StereoBlockMatching::~StereoBlockMatching(){
    // Destructor
}

void StereoBlockMatching::setParameters(int num_disparities, int block_size, cv::Size image_size){
    this->num_disparities = num_disparities;
    this->block_size = block_size;
    this->padding = block_size/2;
    this->disparity_image = cv::Mat::zeros(image_size, CV_32F);
    ref_window_stack.reserve(num_disparities);
    target_window_stack.reserve(num_disparities);
    ssd_values.reserve(num_disparities);
    for(int i = 0; i < image_size.width - padding - 1; i++){
        sliding_windows.emplace_back(num_sliding_windows_v2(image_size.width, i + padding + 1));
    }
}

void StereoBlockMatching::makeGrayscale(cv::Mat &img, cv::Mat &gray_img){
    cv::Mat red, green, blue; 
    cv::extractChannel(img, red, 2);
    cv::extractChannel(img, green, 1);
    cv::extractChannel(img, blue, 0);
    gray_img = 0.299*red + 0.587*green + 0.114*blue;
}

void StereoBlockMatching::addPadding(cv::Mat &img, cv::Mat &padded_img){
    cv::copyMakeBorder(img, padded_img, padding, padding, padding, padding, cv::BORDER_CONSTANT, 0);
}

std::vector<int> StereoBlockMatching::num_sliding_windows_v2(int n_cols, int col) {
    return std::vector<int>{num_disparities - std::max(0, (num_disparities - col + padding)), num_disparities - std::max(0, (num_disparities + col - n_cols))};
}

int StereoBlockMatching::num_sliding_windows(int n_cols, int col, std::string disparity_type) {
    int num_window_reducer = 0;

    if (disparity_type == "left") {
        if (col - num_disparities - padding <= 0) {
            num_window_reducer = num_disparities - col + padding;
        }
    }

    if (disparity_type == "right") {
        if (n_cols - col - num_disparities <= 0) {
            num_window_reducer = num_disparities + col - n_cols;
        }
    }

    int sliding_windows = num_disparities - num_window_reducer;
    return sliding_windows;
}

void StereoBlockMatching::target_window_stacker(cv::Mat &target_padded, int col, int row, std::string disparity_type, int sliding_windows){
    for (int i = 0; i < sliding_windows; i++) {
        if (disparity_type == "left") {
            target_window_stack.emplace_back(target_padded(cv::Rect(col - padding - i, row - padding, block_size, block_size)));
        }
        if (disparity_type == "right") {
            target_window_stack.emplace_back(target_padded(cv::Rect(col - padding + i, row - padding, block_size, block_size)));
        }
    }
}
void StereoBlockMatching::computeDisparityMap(cv::Mat& ref_image, cv::Mat& target_image, cv::Mat &disp){

    cv::Mat ref_gray, target_gray, ref_padded, target_padded;
    makeGrayscale(ref_image, ref_gray);
    makeGrayscale(target_image, target_gray);

    addPadding(ref_gray, ref_padded);
    addPadding(target_gray, target_padded);

    cv::Size size = ref_gray.size();

    int width = size.width;
    int height = size.height;
    auto start = std::chrono::high_resolution_clock::now();
    for(int row = padding +1; row < height; row++){
        for(int col = padding + 1; col < width - padding; col++){
            int num_windows = sliding_windows[col - padding - 1][0];
            auto ref_window = ref_padded(cv::Rect(col - padding, row - padding, block_size, block_size));
            ref_window_stack.clear();
            target_window_stack.clear();
            ssd_values.clear();
            std::fill_n(std::back_inserter(ref_window_stack), num_windows, ref_window);
            target_window_stacker(target_padded, col, row, "left", num_windows);
            for(int i = 0; i < num_windows; i++){
                cv::Mat diff;
                cv::absdiff(ref_window_stack[i], target_window_stack[i], diff);
                cv::pow(diff, 2, diff);
                ssd_values[i] = cv::sum(diff)[0];
            }
            int min_ssd = *std::min_element(ssd_values.begin(), ssd_values.end());
            disp.at<float>(row, col) = min_ssd;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;


    cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8U);
}

int main(){

    cv::Mat left_image = cv::imread("/Users/nitishsanghi/Documents/Stereo-Matching/data_scene_flow/training/image_2/000000_10.png", cv::IMREAD_COLOR);
    cv::Mat right_image = cv::imread("/Users/nitishsanghi/Documents/Stereo-Matching/data_scene_flow/training/image_3/000000_10.png", cv::IMREAD_COLOR);

    std::cout << "Image size: " << left_image.size() << std::endl;
    std::cout << "Image channels: " << left_image.channels() << std::endl;

    StereoBlockMatching sbm;
    sbm.setParameters(100,15,left_image.size());
    cv::Mat disparity_image = cv::Mat::zeros(left_image.size(), CV_32F);
    
    sbm.computeDisparityMap(left_image, right_image, disparity_image);

    cv::imshow("Disparity", disparity_image);
    cv::waitKey(0);
    return 0;
}