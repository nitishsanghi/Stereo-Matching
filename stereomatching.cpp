// Include necessary header files from OpenCV and standard libraries
#include "stereomatching.hpp"

// Set the parameters required for computing the disparity map.
void StereoBlockMatching::setParameters(int num_disparities, int block_size, cv::Size image_size){
    this->num_disparities = num_disparities;
    this->block_size = block_size;
    // Padding is half the block size, used to avoid boundary issues.
    this->padding = block_size/2;
    // Initialize the disparity image with zeros.
    this->disparity_image = cv::Mat::zeros(image_size, CV_32F);
    // Prepare the sliding windows size for each column in advance.
    for(int i = 0; i < image_size.width - padding - 1; i++){
        sliding_windows.emplace_back(num_sliding_windows(image_size.width, i + padding + 1));
    }
}

// Calculate the maximum number of sliding windows for a given column in the image.
int StereoBlockMatching::num_sliding_windows(int n_cols, int col) {
    // Adjust the number of windows based on the column index and image properties.
    return num_disparities - std::max(0, (num_disparities - col + padding));
}

cv::Mat StereoBlockMatching::mean(cv::Mat &window){
    cv::Scalar mean = cv::mean(window);
    cv::Mat mean_mat(window.size(), window.type(), mean);
    return mean_mat;
}

float StereoBlockMatching::ssd(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff){
    cv::absdiff(ref_window, target_window, diff);
    return cv::norm(diff, cv::NORM_L2SQR);
}

float StereoBlockMatching::disparity_sad(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff){
    cv::absdiff(ref_window, target_window, diff);
    return cv::sum(diff)[0];
}
float StereoBlockMatching::disparity_zsad(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff){
    cv::absdiff(ref_window - mean(ref_window), target_window - mean(target_window), diff);
    return cv::sum(diff)[0];
}

float StereoBlockMatching::disparity_ssd(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff){
    cv::absdiff(ref_window, target_window, diff);
    return cv::norm(diff, cv::NORM_L2SQR);
}

float StereoBlockMatching::disparity_ncc(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff){
    cv::Mat result;
    cv::multiply(ref_window, target_window, result);
    cv::Mat denominator_1;
    cv::multiply(ref_window, ref_window, denominator_1);
    cv::Mat denominator_2;
    cv::multiply(target_window, target_window, denominator_2);
    //cv::Mat zero_mat = cv::Mat::zeros(ref_window.size(), ref_window.type());
    return cv::sum(result)[0]/(cv::sum(denominator_1)[0]*cv::sum(denominator_2)[0]);
}


float StereoBlockMatching::disparity_zncc(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff){
    cv::Mat ref_window_mean = ref_window - mean(ref_window);
    cv::Mat target_window_mean = target_window - mean(target_window);
    cv::Mat result;
    cv::multiply(ref_window_mean, target_window_mean, result);
    cv::Mat denominator_1;
    cv::multiply(ref_window_mean, ref_window_mean, denominator_1);
    cv::Mat denominator_2;
    cv::multiply(target_window_mean, target_window_mean, denominator_2);
    //cv::Mat zero_mat = cv::Mat::zeros(ref_window.size(), ref_window.type());
    return cv::sum(result)[0]*cv::sum(result)[0]/(cv::sum(denominator_1)[0]*cv::sum(denominator_2)[0]);
}

// Compute the disparity map by comparing blocks of the two images and finding the best match.
void StereoBlockMatching::computeDisparityMap(cv::Mat& ref_image, cv::Mat& target_image, cv::Mat &disp, DisparityMetric metric){
    // Convert the images to grayscale to simplify the matching process.
    cv::Mat ref_gray, target_gray;
    cv::cvtColor(ref_image, ref_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(target_image, target_gray, cv::COLOR_BGR2GRAY);
    // Add padding to handle the border of the images.
    cv::Mat ref_padded, target_padded;
    cv::copyMakeBorder(ref_gray, ref_padded, padding, padding, padding, padding, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(target_gray, target_padded, padding, padding, padding, padding, cv::BORDER_REPLICATE);

    // Start timing the disparity computation.
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat diff;
    int disparity = 0;
    float ssd_values;
    // Iterate through each pixel in the padded reference image.
    for(int row = padding +1; row < ref_padded.rows - padding; row++){
        for(int col = padding + 1; col < ref_padded.cols - padding; col++){
            // Retrieve the number of windows to slide based on the current column.
            int num_windows = sliding_windows[col - padding - 1];
            // Define the reference block for comparison.
            cv::Mat ref_window = ref_padded(cv::Rect(col - padding, row - padding, block_size, block_size));
            float min_ssd = std::numeric_limits<float>::max();

            // Slide the window across the target image and calculate the Sum of Squared Differences (SSD).

            for(int i = 0; i < num_windows; i++){
                cv::Mat target_window = target_padded(cv::Rect(col - padding - i, row - padding, block_size, block_size));
                if (metric == SAD)
                    ssd_values = disparity_sad(ref_window, target_window, diff);
                if (metric == ZSAD)
                    ssd_values = disparity_zsad(ref_window, target_window, diff);
                if (metric == SSD)
                    ssd_values = disparity_ssd(ref_window, target_window, diff);
                if (metric == NCC)
                    ssd_values = disparity_ncc(ref_window, target_window, diff);
                if (metric == ZNCC)
                    ssd_values = disparity_zncc(ref_window, target_window, diff);
                
                if(ssd_values < min_ssd){
                    min_ssd = ssd_values;
                    disparity = i;
                }
            }


            // Assign the best matching disparity value to the output disparity map.
            disp.ptr<float>(row - padding)[col - padding] = disparity;
        }
    }

    // Normalize the disparity values for visualization.
    cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8U);

    // End timing and print the elapsed time.
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;
}