#ifndef COORDINATE_TRANSFORM
#define COORDINATE_TRANSFORM

#include <iostream>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &K);

Eigen::Vector3d project2Dto3D(double x, double y, double d, Eigen::Matrix3d K, double depth_scale);

inline Eigen::Vector2d project3Dto2D(double x, double y, double z, Eigen::Matrix3d K){
	double u = K(0, 0) * x / z + K(0, 2);
	double v = K(1, 1) * y / z + K(1, 2);
	return Eigen::Vector2d(u, v);
}

#endif