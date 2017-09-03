#ifndef ICP_BA_CERES
#define ICP_BA_CERES

#include <iostream>
#include <ceres/rotation.h>
#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

void ICP_BA_ceres(const std::vector<cv::Point3f>& pts1,
			 const std::vector<cv::Point3f>& pts2,
			 cv::Mat& R,
			 cv::Mat& t);

#endif // ICP_BA_CERES

