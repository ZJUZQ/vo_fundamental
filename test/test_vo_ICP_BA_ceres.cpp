#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "../include/feature_match.hpp"
#include "../include/coordinate_transform.hpp"
#include "../include/ICP_SVD.hpp"
#include "../include/ICP_BA_ceres.hpp"

int main(int argc, char** argv){
	if(argc != 5){
		cout << "usage: ./test_vo_ICP_SVD img1 img2 depth1 depth2" << endl;
		return 1;
	}

	cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);
	cv::Mat depth_1 = cv::imread(argv[3], cv::IMREAD_UNCHANGED);
	cv::Mat depth_2 = cv::imread(argv[4], cv::IMREAD_UNCHANGED);
	cv::Mat K = (cv::Mat_<double> (3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	std::vector<cv::Point3f> pts1;
	std::vector<cv::Point3f> pts2;

	// keypoint match
	std::vector<cv::KeyPoint> kps_1, kps_2;
	std::vector<cv::DMatch> matches;
	find_feature_matches(img_1, img_2, kps_1, kps_2, matches);
	
	for(int i = 0; i < matches.size(); i++){
		unsigned short d1 = depth_1.ptr<unsigned short>( (int)kps_1[matches[i].queryIdx].pt.y )[
			(int)kps_1[matches[i].queryIdx].pt.x ];
		unsigned short d2 = depth_2.ptr<unsigned short>( (int)kps_2[matches[i].trainIdx].pt.y )[
			(int)kps_2[matches[i].trainIdx].pt.x ];

		if(d1 == 0 || d2 == 0) // bad depth
			continue;
		float dd1 = d1 / 1000.0;
		float dd2 = d2 / 1000.0;
		cv::Point2f p1 = pixel2cam(kps_1[matches[i].queryIdx].pt, K);
		cv::Point2f p2 = pixel2cam(kps_2[matches[i].trainIdx].pt, K);
		pts1.push_back( cv::Point3f(p1.x * dd1, p1.y * dd1, dd1) );
		pts2.push_back( cv::Point3f(p2.x * dd2, p2.y * dd2, dd2) );
	}

	cout << "3d-3d pairs: " << pts1.size() << endl;

	cv::Mat R, t;

	ICP_SVD(pts1, pts2, R, t); // pts1 = R * pts2 + t

	cout << "ICP via SVD results: " << endl;
	cout << "R = \n" << R << endl;
	cout << "t = \n" << t << endl;

	ICP_BA_ceres(pts1, pts2, R, t);

	cout << "ICP via BA_ceres results: " << endl;
	cout << "R = \n" << R << endl;
	cout << "t = \n" << t << endl;

	return 0;
}