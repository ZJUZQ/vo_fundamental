#include "../include/3d3d_SVD.hpp"

void _3d3d_SVD(const std::vector<cv::Point3d>& pts1,
			 const std::vector<cv::Point3d>& pts2,
			 cv::Mat& R,
			 cv::Mat& t){

	cv::Point3d p1_c, p2_c; // center of mass
	int N = pts2.size();
	for(int i = 0; i < N; i++){
		p1_c += pts1[i];
		p2_c += pts2[i];
	}
	p1_c /= N;
	p2_c /= N;

	std::vector<cv::Point3d> pts1_r(N), pts2_r(N); // coordinate removed the mass center
	for(int i = 0; i < N; i++){
		pts1_r[i] = pts1[i] - p1_c;
		pts2_r[i] = pts2[i] - p2_c;
	}
	Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
	for(int i = 0; i < N; i++){
		W += Eigen::Vector3d(pts1_r[i].x, pts1_r[i].y, pts1_r[i].z) * (Eigen::Vector3d(
			pts2_r[i].x, pts2_r[i].y, pts2_r[i].z).transpose());
	}
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	std::cout << "U = \n" << U << std::endl;
	std::cout << "v = \n" << V << std::endl;

	Eigen::Matrix3d _R = U * (V.transpose());
	Eigen::Vector3d _t = Eigen::Vector3d(p1_c.x, p1_c.y, p1_c.z) - _R * Eigen::Vector3d(
						 p2_c.x, p2_c.y, p2_c.z);

	//convert to cv::Mat
	// pts1 = R * pts2 + t
	R = (cv::Mat_<double>(3, 3) <<
		_R(0, 0), _R(0, 1), _R(0, 2),
		_R(1, 0), _R(1, 1), _R(1, 2),
		_R(2, 0), _R(2, 1), _R(2, 2));
	t = ( cv::Mat_<double>(3, 1) << _t(0, 0), _t(1, 0), _t(2, 0) );
}
