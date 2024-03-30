# StereoBlockMatching: A Manual Stereo Matching Implementation

## Introduction
The `StereoBlockMatching` class is a manual implementation of stereo block matching for computing disparity maps from stereo images. This approach provides a clear understanding of the stereo matching process using basic OpenCV functions, allowing for a customizable and insightful exploration into stereo vision.
![alt text](https://github.com/nitishsanghi/Stereo-Matching/blob/main/000000_10.png)
![alt text](https://github.com/nitishsanghi/Stereo-Matching/blob/main/Disparity.png)

## Features
- **Customizable Disparity Metrics**: The implementation supports various methods for calculating disparity, such as Sum of Absolute Differences (SAD), Zero-mean Sum of Absolute Differences (ZSAD), Sum of Squared Differences (SSD), Normalized Cross-Correlation (NCC), and Zero-mean Normalized Cross-Correlation (ZNCC).
- **Manual Implementation**: Leverages fundamental OpenCV functions to illustrate the stereo matching algorithm's core principles.
- **Performance Insights**: Includes timing for the disparity computation to analyze and optimize performance.

## Dependencies
- [OpenCV](https://opencv.org/) (4.x recommended)
- C++11 or later

## Installation
Ensure OpenCV is installed and configured in your C++ development environment. Clone the repository or copy the `StereoBlockMatching.hpp` file into your project directory.

## Usage

### Initialization
Instantiate the `StereoBlockMatching` object and set the stereo matching parameters.

```cpp
StereoBlockMatching sbm;
sbm.setParameters(60, 9, cv::Size(640, 480)); // Parameters: num_disparities, block_size, image_size
```

### Compute the Disparity Map
Select the disparity metric and compute the disparity map using left and right stereo images.

```cpp
cv::Mat leftImage = cv::imread("left.jpg", cv::IMREAD_GRAYSCALE);
cv::Mat rightImage = cv::imread("right.jpg", cv::IMREAD_GRAYSCALE);
cv::Mat disparityMap;

sbm.computeDisparityMap(leftImage, rightImage, disparityMap, StereoBlockMatching::SSD); // Using SSD as the metric
```

### Visualization
Normalize and display the computed disparity map for visualization.

```cpp
cv::normalize(disparityMap, disparityMap, 0, 255, cv::NORM_MINMAX, CV_8U);
cv::imshow("Disparity Map", disparityMap);
cv::waitKey(0);
```

## Disparity Metrics and Equations

### Sum of Absolute Differences (SAD)
<img src="https://render.githubusercontent.com/render/math?math=\text{SAD}(I_1, I_2) = \sum_{x, y} |I_1(x, y) - I_2(x, y)|">

### Zero-mean Sum of Absolute Differences (ZSAD)
\[ \text{ZSAD}(I_1, I_2) = \sum_{x, y} |(I_1(x, y) - \mu_{I_1}) - (I_2(x, y) - \mu_{I_2})| \]

### Sum of Squared Differences (SSD)
\[ \text{SSD}(I_1, I_2) = \sum_{x, y} (I_1(x, y) - I_2(x, y))^2 \]

### Normalized Cross-Correlation (NCC)
\[ \text{NCC}(I_1, I_2) = \frac{\sum_{x, y} I_1(x, y) \times I_2(x, y)}{\sqrt{\sum_{x, y} I_1(x, y)^2 \times \sum_{x, y} I_2(x, y)^2}} \]

### Zero-mean Normalized Cross-Correlation (ZNCC)
\[ \text{ZNCC}(I_1, I_2) = \frac{\sum_{x, y} (I_1(x, y) - \mu_{I_1}) \times (I_2(x, y) - \mu_{I_2})}{\sqrt{\sum_{x, y} (I_1(x, y) - \mu_{I_1})^2 \times \sum_{x, y} (I_2(x, y) - \mu_{I_2})^2}} \]

## Contributing
Contributions to improve the algorithm or extend its functionality are welcome. Open an issue or pull request with your suggestions or improvements.

## Performance Considerations
While focusing on clarity, this manual implementation may not match the performance of optimized OpenCV functions. Consider parallel processing or algorithmic optimizations for real-time applications.

## License
This project is open-sourced under the MIT License. See the LICENSE file for details.

---

This write-up, enriched with equations, provides a comprehensive guide to using and understanding the `StereoBlockMatching` class, making it accessible for educational purposes and practical applications in stereo vision projects.
