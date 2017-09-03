#ifndef ICP_SVD_
#define ICP_SVD_

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

void ICP_SVD(const std::vector<cv::Point3f>& pts1,
			 const std::vector<cv::Point3f>& pts2,
			 cv::Mat& R,
			 cv::Mat& t);

#endif