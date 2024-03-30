// Include necessary header files from OpenCV and standard libraries
#include "stereomatching.hpp"

// Sets the necessary parameters for the disparity computation process.
void StereoBlockMatching::setParameters(int num_disparities, int block_size, cv::Size image_size) {
    this->num_disparities = num_disparities; // The maximum disparity difference to consider for matching.
    this->block_size = block_size; // The size of the square block to use for block matching.
    this->padding = block_size / 2; // Padding to be added to the image borders to handle edge cases.
    
    // Initialize the disparity image to zero. It will hold the result of the disparity computation.
    this->disparity_image = cv::Mat::zeros(image_size, CV_32F);
    
    // Precompute the number of sliding windows for each possible column, considering the given padding.
    sliding_windows.clear(); // Clearing any previously stored values.
    for (int i = 0; i < image_size.width - padding - 1; i++) {
        sliding_windows.emplace_back(num_sliding_windows(image_size.width, i + padding + 1));
    }
}

// Calculates the number of sliding windows that can fit within the current column, considering the padding and disparity range.
int StereoBlockMatching::num_sliding_windows(int n_cols, int col) {
    return num_disparities - std::max(0, (num_disparities - col + padding));
}

// Calculates the mean value of the pixels within a window. This is used in zero-mean metrics.
cv::Mat StereoBlockMatching::mean(cv::Mat &window) {
    cv::Scalar meanVal = cv::mean(window); // Calculate the mean of the window.
    return cv::Mat(window.size(), window.type(), meanVal); // Return a matrix filled with the mean value.
}

// Computes the Sum of Squared Differences (SSD) between two image windows.
float StereoBlockMatching::ssd(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff) {
    cv::absdiff(ref_window, target_window, diff); // Calculate the absolute difference between the windows.
    return cv::norm(diff, cv::NORM_L2SQR); // Return the squared L2 norm of the difference.
}

// Computes the Sum of Absolute Differences (SAD) between two image windows.
float StereoBlockMatching::disparity_sad(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff) {
    cv::absdiff(ref_window, target_window, diff); // Calculate the absolute difference between the windows.
    return cv::sum(diff)[0]; // Sum up all the differences to get the SAD.
}

// Computes the Zero-mean Sum of Absolute Differences (ZSAD) between two image windows.
float StereoBlockMatching::disparity_zsad(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff){
    cv::absdiff(ref_window_mean, target_window - mean(target_window), diff); // Subtract means and calculate the absolute difference.
    return cv::sum(diff)[0]; // Sum up all the differences to get the ZSAD.
}

float StereoBlockMatching::disparity_ssd(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff){
    cv::absdiff(ref_window, target_window, diff);
    return cv::norm(diff, cv::NORM_L2SQR);
}

float StereoBlockMatching::disparity_ncc(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff){
    cv::multiply(ref_window, target_window, cross_cor_mat);
    cv::multiply(target_window, target_window, target_ssd_mat);
    return cv::sum(cross_cor_mat)[0]/(ref_ssd*cv::sum(target_ssd_mat)[0]);
}


float StereoBlockMatching::disparity_zncc(cv::Mat &ref_window, cv::Mat &target_window, cv::Mat &diff){
    target_window_mean = target_window - mean(target_window);
    cv::multiply(ref_window_mean, target_window_mean, cross_cor_mat);
    cv::multiply(target_window_mean, target_window_mean, target_ssd_mat);
    cross_cor = cv::sum(cross_cor_mat)[0];
    return cross_cor*cross_cor/(ref_ssd*cv::sum(target_ssd_mat)[0]);
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
            num_windows = sliding_windows[col - padding - 1];
            // Define the reference block for comparison.
            ref_window = ref_padded(cv::Rect(col - padding, row - padding, block_size, block_size));
            min_ssd = MIN_SSD_CONSTANT;

            // Slide the window across the target image and calculate the Sum of Squared Differences (SSD).
            switch(metric){
                case SAD:
                        break;
                case ZSAD:
                    ref_window_mean = ref_window - mean(ref_window);
                    break;
                case SSD:
                    break;
                case NCC:
                    cv::multiply(ref_window, ref_window, ref_ssd_mat);
                    ref_ssd = cv::sum(ref_ssd_mat)[0];
                    break;
                
                case ZNCC:
                    ref_window_mean = ref_window - mean(ref_window);
                    cv::multiply(ref_window_mean, ref_window_mean, ref_ssd_mat);
                    ref_ssd = cv::sum(ref_ssd_mat)[0];
                    break;
            }

            for(int i = 0; i < num_windows; i++){
                target_window = target_padded(cv::Rect(col - padding - i, row - padding, block_size, block_size));
                
                switch(metric){
                    case SAD:
                        ssd_values = disparity_sad(ref_window, target_window, diff);
                        break;
                    case ZSAD:
                        ssd_values = disparity_zsad(ref_window, target_window, diff);
                        break;
                    case SSD:
                        ssd_values = disparity_ssd(ref_window, target_window, diff);
                        break;
                    case NCC:
                        ssd_values = disparity_ncc(ref_window, target_window, diff);
                        break;        
                    case ZNCC:
                        ssd_values = disparity_zncc(ref_window, target_window, diff);
                        break;
                }
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