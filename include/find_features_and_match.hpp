#ifndef FIND_FEATURES_AND_MATCH_H
#define FIND_FEATURES_AND_MATCH_H

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace std;

void find_features_and_match(cv::Mat &img_1, 
							 cv::Mat &img_2,
						     std::vector<cv::KeyPoint> &kps_1,
						     std::vector<cv::KeyPoint> &kps_2,
						     std::vector<cv::DMatch> &matches);

#endif