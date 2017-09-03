
#include "../include/bundle_adjustment_ceres.hpp"

struct ReprojectionError {
	ReprojectionError(double observed_x, double observed_y, cv::Mat K) 
		: _observed_x(observed_x), _observed_y(observed_y), _K(K) {}

	template <typename T>
	bool operator()(const T* const pose,
	      		const T* const point_3d,
	      		T* residuals) const {
		// pose[0,1,2] are the angle-axis rotation.
		T p[3];

		// p = R(angle_axis) * point_3d;
		ceres::AngleAxisRotatePoint(pose, point_3d, p); //#include "ceres/rotation.h"

		// pose[3,4,5] are the translation.
		p[0] += pose[3]; p[1] += pose[4]; p[2] += pose[5];

		T predicted_x = _K.at<T>(0, 0) * p[0] / p[2] + _K.at<T>(0, 2);
		T predicted_y = _K.at<T>(1, 1) * p[1] / p[2] + _K.at<T>(1, 2);

		// The error is the difference between the predicted and observed position.
		residuals[0] = predicted_x - T(_observed_x);
		residuals[1] = predicted_y - T(_observed_y);
		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(const double observed_x,
	                          const double observed_y, cv::Mat K){
		return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
		         new ReprojectionError(observed_x, observed_y, K)));
	}

	double _observed_x;
	double _observed_y;
	cv::Mat _K; // camera matrix
};

void bundle_adjustment_ceres(const std::vector<cv::Point3f> pts_3d,
							 const std::vector<cv::Point2f> pts_2d,
							 const cv::Mat& K,
							 cv::Mat R, cv::Mat t){ // pose: angleaxis, translate
	
	//	Each residual in a BAL problem depends on a three dimensional: point_3d;
	//	and a six parameter: pose. 

	//	The six parameters defining the pose are: three for rotation as a Rodrigues’ 
	//	axis-angle vector, three for translation.
	
	cv::Mat phi;
	cv::Rodrigues(R, phi); // rotation matrix to angleaxis
	double camera_pose[6] = {phi.at<double>(0, 0),
							phi.at<double>(1, 0),
							phi.at<double>(2, 0),
							t.at<double>(0, 0),
							t.at<double>(1, 0),
							t.at<double>(2, 0)};

	double pts_3d_array[3 * pts_3d.size()];

	ceres::Problem problem;
	for (int i = 0; i < pts_3d.size(); ++i) {
		ceres::CostFunction* cost_function = ReprojectionError::Create(pts_2d[i].x, pts_2d[i].y, K);

		pts_3d_array[3 * i] = pts_3d[i].x;
		pts_3d_array[3 * i + 1] = pts_3d[i].y;
		pts_3d_array[3 * i + 2] = pts_3d[i].z;

		problem.AddResidualBlock(cost_function,
		                       NULL , //squared loss 
		                       camera_pose,
		                       pts_3d_array + 3 * i);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.gradient_tolerance = 1e-16;
  	options.function_tolerance = 1e-16;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	phi = (cv::Mat_<double>(3, 1) << camera_pose[0], camera_pose[1], camera_pose[2]);
	cv::Rodrigues(phi, R);
	t = (cv::Mat_<double>(3, 1) << camera_pose[3], camera_pose[4], camera_pose[5]);
}



/*
#include "../include/bundle_adjustment_ceres.hpp"

struct ReprojectionError {
	ReprojectionError(double observed_x, double observed_y, cv::Mat K, cv::Point3f pt_3d) 
		: _observed_x(observed_x), _observed_y(observed_y), _K(K), _point_3d(pt_3d) {}

	template <typename T>
	bool operator()(const T* const pose,
	      			T* residuals) const {
		// pose[0,1,2] are the angle-axis rotation.
		T p[3];

		// p = R(angle_axis) * point_3d;
		T point_3d[3] = {T(_point_3d.x), T(_point_3d.y), T(_point_3d.z)};
		ceres::AngleAxisRotatePoint(pose, point_3d, p); //#include "ceres/rotation.h"

		// pose[3,4,5] are the translation.
		p[0] += pose[3]; p[1] += pose[4]; p[2] += pose[5];

		T predicted_x = _K.at<T>(0, 0) * p[0] / p[2] + _K.at<T>(0, 2);
		T predicted_y = _K.at<T>(1, 1) * p[1] / p[2] + _K.at<T>(1, 2);

		// The error is the difference between the predicted and observed position.
		residuals[0] = predicted_x - T(_observed_x);
		residuals[1] = predicted_y - T(_observed_y);
		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(const double observed_x,
	                          const double observed_y, cv::Mat K, cv::Point3f pt_3d){
		return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(
		         new ReprojectionError(observed_x, observed_y, K, pt_3d)));
	}

	double _observed_x;
	double _observed_y;
	cv::Mat _K; // camera matrix
	cv::Point3f _point_3d;
};

void bundle_adjustment_ceres(const std::vector<cv::Point3f> pts_3d,
							 const std::vector<cv::Point2f> pts_2d,
							 const cv::Mat& K,
							 cv::Mat R, cv::Mat t){ // pose: angleaxis, translate
	
	//	Each residual in a BAL problem depends on a three dimensional: point_3d;
	//	and a six parameter: pose. 

	//	The six parameters defining the pose are: three for rotation as a Rodrigues’ 
	//	axis-angle vector, three for translation.
	
	cv::Mat phi;
	cv::Rodrigues(R, phi); // rotation matrix to angleaxis
	double camera_pose[6] = {phi.at<double>(0, 0),
							phi.at<double>(1, 0),
							phi.at<double>(2, 0),
							t.at<double>(0, 0),
							t.at<double>(1, 0),
							t.at<double>(2, 0)};

	

	ceres::Problem problem;
	for (int i = 0; i < pts_3d.size(); ++i) {
		ceres::CostFunction* cost_function = ReprojectionError::Create(pts_2d[i].x, pts_2d[i].y, K, pts_3d[i]);

		problem.AddResidualBlock(cost_function,
		                       NULL , //squared loss 
		                       camera_pose);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.gradient_tolerance = 1e-16;
  	options.function_tolerance = 1e-16;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	phi = (cv::Mat_<double>(3, 1) << camera_pose[0], camera_pose[1], camera_pose[2]);
	cv::Rodrigues(phi, R);
	t = (cv::Mat_<double>(3, 1) << camera_pose[3], camera_pose[4], camera_pose[5]);
}
*/



