#include "../include/pose_estimation_2d2d.hpp"

void pose_estimation_2d2d(std::vector<cv::KeyPoint> kps_1, 
						  std::vector<cv::KeyPoint> kps_2, 
						  std::vector<cv::DMatch> matches, 
						  cv::Mat& R, cv::Mat& t){
	cv::Mat K = (cv::Mat_<double> (3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	vector<cv::Point2f> pts_1, pts_2;
	for(int i = 0; i < (int)matches.size(); i++){
		pts_1.push_back(kps_1[matches[i].queryIdx].pt);
		pts_2.push_back(kps_2[matches[i].trainIdx].pt);
	}
	cv::Mat F; //fundamental matrix
	F = cv::findFundamentalMat(pts_1, pts_2, 2);
	cout << "fundamental matrix is: \n" << F << endl;

	cv::Point2d principal_point(325.1, 249.7);
	int focal_length = 521;
	cv::Mat E; // essential matrix
	E = cv::findEssentialMat(pts_1, pts_2, focal_length, principal_point, cv::RANSAC);
	cout << "essential matrix is: \n" << E << endl;

	cv::Mat H; // homograph matrix
	H = cv::findHomography(pts_1, pts_2, cv::RANSAC, 3, cv::noArray(), 2000, 0.99);
	cout << "homograph matrix is: \n" << H << endl;

	cv::recoverPose(E, pts_1, pts_2, R, t, focal_length, principal_point);
	cout << "R is: \n" << R << endl;
	cout << "t is: \n" << t << endl;
}
