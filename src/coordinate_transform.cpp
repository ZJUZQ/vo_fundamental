#include "../include/coordinate_transform.hpp"

cv::Point2f pixel2cam(const cv::Point2f &p, const cv::Mat &K){
	double X = (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0);
	double Y = (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1);
	return cv::Point2f(X, Y);
}