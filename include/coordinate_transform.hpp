#ifndef COORDINATE_TRANSFORM
#define COORDINATE_TRANSFORM

#include <iostream>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &K);

#endif