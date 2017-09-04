#ifndef _3D2D_BA_CERES_H
#define _3D2D_BA_CERES_H

#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

void bundle_adjustment_ceres(const std::vector<cv::Point3f> pts_3d,
							 const std::vector<cv::Point2f> pts_2d,
							 const cv::Mat& K,
							 cv::Mat R, cv::Mat t);

#endif