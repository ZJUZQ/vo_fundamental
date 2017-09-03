#ifndef TRIANGULATION
#define TRIANGULATION

#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

void triangulation(const vector<cv::KeyPoint>& kps_1,
				   const vector<cv::KeyPoint>& kps_2,
				   const vector<cv::DMatch>& matches,
				   const cv::Mat& R, const cv::Mat& t,
				   vector<cv::Point3d>& points_3d);

#endif