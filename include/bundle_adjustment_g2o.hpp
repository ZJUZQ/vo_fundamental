#ifndef BUNDLE_ADJUSTMENT_G2O_H
#define BUNDLE_ADJUSTMENT_G2O_H

#include <iostream>
#include <vector>
#include <ctime>

#include <opencv2/core/core.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;

void bundle_adjustment_g2o(const std::vector<cv::Point3f> pts_3d,
						   const std::vector<cv::Point2f> pts_2d,
						   const cv::Mat& K,
						   cv::Mat R, 
						   cv::Mat t);

#endif