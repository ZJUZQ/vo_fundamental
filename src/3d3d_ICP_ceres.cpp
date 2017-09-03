#include "../include/3d3d_ICP_ceres.hpp"

struct ReprojectionError_ICP_BA{

	ReprojectionError_ICP_BA(double x, double y, double z)
					:pt1_x(x), pt1_y(y), pt1_z(z) {}

	template <typename T>
	bool operator()(const T* const se3_,  // angleaxis + t
					const T* const pt2_3d,
					T* residuals) const{
		T p1[3];
		ceres::AngleAxisRotatePoint(se3_, pt2_3d, p1);
		p1[0] += se3_[3];
		p1[1] += se3_[4];
		p1[2] += se3_[5];

		// the error is the difference pt1_3d - (R * pt2_3d + t)
		residuals[0] = T(pt1_x) - p1[0];
		residuals[1] = T(pt1_y) - p1[1];
		residuals[2] = T(pt1_z) - p1[2];
		return true;
	}

private:
	//observation for a sample
	const double pt1_x;
	const double pt1_y;
	const double pt1_z;
};

void _3d3d_ICP_ceres(const std::vector<cv::Point3d>& pts1,
			 const std::vector<cv::Point3d>& pts2,
			 cv::Mat& R,
			 cv::Mat& t){

	// initial the optimized parameters
	double se3[6];
	cv::Mat phi;
	cv::Rodrigues(R, phi);
	se3[0] = phi.at<double>(0, 0);
	se3[1] = phi.at<double>(1, 0);
	se3[2] = phi.at<double>(2, 0);
	se3[3] = t.at<double>(0, 0);
	se3[4] = t.at<double>(1, 0);
	se3[5] = t.at<double>(2, 0);

	double pts2_3d[3 * pts2.size()];
	for(int i = 0; i < pts2.size(); i++){
		pts2_3d[3 * i] = pts2[i].x;
		pts2_3d[3 * i + 1] = pts2[i].y;
		pts2_3d[3 * i + 2] = pts2[i].z;
	}

	ceres::Problem problem;
	for(int i = 0; i < pts2.size(); i++){
		ceres::CostFunction* cost_func =
			new ceres::AutoDiffCostFunction<ReprojectionError_ICP_BA, 3, 6, 3>(new ReprojectionError_ICP_BA(pts1[i].x, pts1[i].y, pts1[i].z));
		problem.AddResidualBlock(cost_func, NULL, se3, pts2_3d + 3 * i);
	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << std::endl;

	phi = (cv::Mat_<double>(3, 1) << se3[0], se3[1], se3[2]);
	cv::Rodrigues(phi, R);
	t = (cv::Mat_<double>(3, 1) << se3[3], se3[4], se3[5]);
}