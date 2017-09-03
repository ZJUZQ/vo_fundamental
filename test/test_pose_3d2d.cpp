#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "../include/find_features_and_match.hpp"
#include "../include/coordinate_transform.hpp"
#include "../include/bundle_adjustment_ceres.hpp"
#include "../include/bundle_adjustment_g2o.hpp"

int main(int argc, char** argv){
	if(argc != 4){
		cout << "usage: ./test_pose_3d2d img1 img2 depth1" << endl;
		return 1;
	}

	cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);
	cv::Mat depth_1 = cv::imread(argv[3], cv::IMREAD_UNCHANGED); 
	cv::Mat K = (cv::Mat_<double> (3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	std::vector<cv::Point3f> pts_3d;
	std::vector<cv::Point2f> pts_2d;

	// keypoint match
	std::vector<cv::KeyPoint> kps_1, kps_2;
	std::vector<cv::DMatch> matches;
	find_features_and_match(img_1, img_2, kps_1, kps_2, matches);
	cout << "total find " << matches.size() << " pairs of match points\n"; 
	
	// 建立3D点
	for(int i = 0; i < matches.size(); i++){
		// 深度图为16位无符号数，单通道图像
		unsigned short d = depth_1.ptr<unsigned short>( (int)kps_1[matches[i].queryIdx].pt.y )[
			(int)kps_1[matches[i].queryIdx].pt.x ];
		if(d == 0) // bad depth
			continue;
		float dd = d / 1000.0;
		cv::Point2f p1 = pixel2cam(kps_1[matches[i].queryIdx].pt, K);
		pts_3d.push_back( cv::Point3f(p1.x * dd, p1.y * dd, dd) );
		pts_2d.push_back(kps_2[matches[i].trainIdx].pt);
	}
	cout << "3d-2d pairs: " << pts_3d.size() << endl;

	cv::Mat r_vec, t; //  rotation vector
	cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r_vec, t, false, cv::SOLVEPNP_EPNP);
	cv::Mat R;
	cv::Rodrigues(r_vec, R); // / r为旋转向量形式，用Rodrigues公式转换为矩阵

	cout << "r_vec = \n" << r_vec << endl;
	cout << "R = \n" << R << endl;
	cout << "t = \n" << t << endl << endl << endl;

	/**
	bundle_adjustment_ceres(pts_3d, pts_2d, K, R, t);

	cout << "### after ceres optimaton ###" << endl;
	cout << "R = \n" << R << endl;
	cout << "t = \n" << t << endl << endl;
	*/

	bundle_adjustment_g2o(pts_3d, pts_2d, K, R, t);

	cout << "### after g2o optimaton ###" << endl;
	cout << "R = \n" << R << endl;
	cout << "t = \n" << t << endl;
}