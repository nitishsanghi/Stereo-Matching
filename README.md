# StereoBlockMatching Implementation 

## Overview
The `StereoBlockMatching` class is a custom implementation of the stereo block matching algorithm for computing disparity maps from stereo image pairs. Unlike leveraging high-level OpenCV stereo matching functions like `StereoBM` or `StereoSGBM`, this class provides a hands-on approach to understanding and customizing stereo matching at a fundamental level. 

## Features
- **Customizable Block Matching**: Offers flexibility in choosing the disparity metric (SSD, SAD, etc.) for matching.
- **Manual Implementation**: Utilizes basic OpenCV functions for a clearer understanding of the stereo matching process.
- **Performance Insights**: Includes timing for disparity map computation, offering insights into the algorithm's performance.

## Dependencies
- OpenCV 4.x
- C++11 or later

## Usage

### Setting Parameters
Before computing the disparity map, set the required parameters: the number of disparities to consider, the block size for matching, and the size of the images being processed.

```cpp
StereoBlockMatching sbm;
sbm.setParameters(64, 9, cv::Size(640, 480)); // Example parameters
```

### Computing the Disparity Map
Choose a disparity metric and compute the disparity map by providing left and right stereo images. The result is stored in a `cv::Mat`.

```cpp
cv::Mat leftImage = cv::imread("left.jpg", cv::IMREAD_GRAYSCALE);
cv::Mat rightImage = cv::imread("right.jpg", cv::IMREAD_GRAYSCALE);
cv::Mat disparityMap;

sbm.computeDisparityMap(leftImage, rightImage, disparityMap, StereoBlockMatching::SSD);
```

### Disparity Metrics
The class supports various disparity metrics, selectable when calling `computeDisparityMap`:
- `SSD`: Sum of Squared Differences
- `SAD`: Sum of Absolute Differences
- `NCC`: Normalized Cross-Correlation (planned)
- `ZNCC`: Zero-Mean Normalized Cross-Correlation (planned)

Currently, SSD and SAD are implemented, with plans to add support for correlation-based metrics.

## Implementation Details
The class defines basic stereo matching operations such as SSD computation, with a straightforward approach to sliding window management and disparity calculation. The choice of metric influences the matching process, enabling a comparison of different matching strategies' effectiveness.

### Extensibility
The architecture allows for easy extension with new disparity metrics or optimizations like parallel processing for performance improvements.

### Performance and Optimization
The manual implementation provides a clear understanding of the stereo matching process but may not match the performance of optimized OpenCV functions out-of-the-box. Profiling and optimization, such as using SIMD instructions or multi-threading, can enhance performance for real-time applications.

## Conclusion
`StereoBlockMatching` is a valuable educational tool for exploring stereo vision fundamentals. It offers a customizable platform for experimenting with different matching strategies and optimizations in stereo vision applications.
