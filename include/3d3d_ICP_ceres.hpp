#ifndef _3D3D_ICP_CERES_H
#define _3D3D_ICP_CERES_H

#include <iostream>
#include <ceres/rotation.h>
#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

void _3d3d_ICP_ceres(const std::vector<cv::Point3d>& pts1,
			 		 const std::vector<cv::Point3d>& pts2,
			 		 cv::Mat& R,
			 		 cv::Mat& t);

#endif 

