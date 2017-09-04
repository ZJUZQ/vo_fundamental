#include "../include/coordinate_transform.hpp"

cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &K){
	double X = (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0);
	double Y = (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1);
	return cv::Point2f(X, Y);
}

Eigen::Vector3d project2Dto3D(double x, double y, double d, Eigen::Matrix3d K, double depth_scale){
	double zz = d / depth_scale;
	double xx = zz * (x - K(0, 2)) / K(0, 0);
	double yy = zz * (y - K(1, 2)) / K(1, 1);
	return Eigen::Vector3d(xx, yy, zz);
}