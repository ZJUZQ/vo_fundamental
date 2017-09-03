#ifndef _3D3D_SVD_H
#define _3D3D_SVD_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

void _3d3d_SVD(const std::vector<cv::Point3d>& pts1,
			 const std::vector<cv::Point3d>& pts2,
			 cv::Mat& R,
			 cv::Mat& t);

#endif