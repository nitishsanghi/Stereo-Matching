#include "stereomatching.hpp"

int main(){

    cv::Mat left_image = cv::imread("/Users/nitishsanghi/Documents/Stereo-Matching/data_scene_flow/training/image_2/000000_10.png", cv::IMREAD_COLOR);
    cv::Mat right_image = cv::imread("/Users/nitishsanghi/Documents/Stereo-Matching/data_scene_flow/training/image_3/000000_10.png", cv::IMREAD_COLOR);

    std::cout << "Image size: " << left_image.size() << std::endl;
    std::cout << "Image channels: " << left_image.channels() << std::endl;

    StereoBlockMatching sbm;
    sbm.setParameters(100,11,left_image.size());
    cv::Mat disparity_image = cv::Mat::zeros(left_image.size(), CV_32F);
    sbm.computeDisparityMap(left_image, right_image, disparity_image, StereoBlockMatching::DisparityMetric::SAD);
    cv::imwrite("Disparity_SAD.png", disparity_image);

    disparity_image = cv::Mat::zeros(left_image.size(), CV_32F);
    sbm.computeDisparityMap(left_image, right_image, disparity_image, StereoBlockMatching::DisparityMetric::ZSAD);
    cv::imwrite("Disparity_ZSAD.png", disparity_image);
    
    disparity_image = cv::Mat::zeros(left_image.size(), CV_32F);
    sbm.computeDisparityMap(left_image, right_image, disparity_image, StereoBlockMatching::DisparityMetric::SSD);
    cv::imwrite("Disparity_SSD.png", disparity_image);
    
    disparity_image = cv::Mat::zeros(left_image.size(), CV_32F);
    sbm.computeDisparityMap(left_image, right_image, disparity_image, StereoBlockMatching::DisparityMetric::NCC);
    cv::imwrite("Disparity_NCC.png", disparity_image);
    
    disparity_image = cv::Mat::zeros(left_image.size(), CV_32F);
    sbm.computeDisparityMap(left_image, right_image, disparity_image, StereoBlockMatching::DisparityMetric::ZNCC);
    cv::imwrite("Disparity_ZNCC.png", disparity_image);

    return 0;
}