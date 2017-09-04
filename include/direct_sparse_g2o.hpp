#ifndef DIRECT_SPARSE_G2O_H
#define DIRECT_SPARSE_G2O_H

#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
using namespace g2o;

// 一次测量的值，包括一个世界坐标系下三维点与一个灰度值
struct Measurement{
	Measurement(Eigen::Vector3d p, double g) : _p_world(p), _grayscale(g) {}
	Eigen::Vector3d _p_world;
	double _grayscale;
};

bool direct_sparse_g2o(const std::vector<Measurement>& measurements, 
					   cv::Mat& gray, 
					   Eigen::Matrix3d& K, 
					   Eigen::Isometry3d& T_cw);

#endif