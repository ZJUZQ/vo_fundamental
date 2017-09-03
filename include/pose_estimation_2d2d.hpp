#ifndef POSE_ESTIMATION_2D2D
#define POSE_ESTIMATION_2D2D

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace std;

void pose_estimation_2d2d(std::vector<cv::KeyPoint> kps_1, 
						  std::vector<cv::KeyPoint> kps_2, 
						  std::vector<cv::DMatch> matches, 
						  cv::Mat& R, 
						  cv::Mat& t);

#endif // POSE_ESTIMATION_2D2D