#ifndef _3D3D_ICP_G2O_H
#define _3D3D_ICP_G2O_H

#include <iostream>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
// #include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;

void _3d3d_ICP_g2o(	const std::vector<cv::Point3d>& pts1,
			 	   	const std::vector<cv::Point3d>& pts2,
			 	    cv::Mat& R,
			 		cv::Mat& t);

#endif //