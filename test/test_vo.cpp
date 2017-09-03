#include <iostream>
using namespace std;

#include "../include/pose_estimation_2d2d.hpp"
#include "../include/feature_match.hpp"
#include "../include/triangulation.hpp"
#include "../include/coordinate_transform.hpp"

int main(int argc, char** argv){
	if(argc != 3){
		cout << "usage: ./test_pose_estimation_2d2d img1 img2" << endl;
		return 1;
	}
	cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);

	vector<cv::KeyPoint> kps_1, kps_2;
	vector<cv::DMatch> good_matches;
	find_feature_matches(img_1, img_2, kps_1, kps_2, good_matches);
	cout << "find total " << good_matches.size() << " matched points" << endl;

	cv::Mat R, t;
	pose_estimation_2d2d(kps_1, kps_2, good_matches, R, t);

	cv::Mat K = (cv::Mat_<double> (3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	/*
	cv::Mat t_hat = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
										   t.at<double>(2, 0), 0, -t.at<double>(0, 0),
										   -t.at<double>(1, 0), t.at<double>(0, 0), 0);
	cout << "t^R = \n" << t_hat * R << endl;

	cv::Mat K = (cv::Mat_<double> (3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	for(int i = 0; i < good_matches.size(); i++){
		cv::Mat pt1 = pixel2cam(kps_1[good_matches[i].queryIdx].pt, K);
		cv::Mat pt2 = pixel2cam(kps_2[good_matches[i].trainIdx].pt, K);
		cv::Mat d = pt2.t() * t_hat * R * pt1;
		cout << "epipolar constraint = " << d << endl;
	}
	*/

	// triangulation
	std::vector<cv::Point3d> points_3d;
	triangulation(kps_1, kps_2, good_matches, R, t, points_3d);

	// verify triangulation, reprojection
	for(int i = 0; i < good_matches.size(); i++){
		cv::Point2d pt1_cam = pixel2cam(kps_1[good_matches[i].queryIdx].pt, K);
		cv::Point2d pt1_cam_3d(
			points_3d[i].x / points_3d[i].z,
			points_3d[i].y / points_3d[i].z);
		//cout << "point in the first camera frame: " << pt1_cam << endl;
		//cout << "point in the first camera frame projected from 3D: " << pt1_cam_3d << endl << endl;
		cout << "d = " << points_3d[i].z << endl;
		cout << "frame_1, diff in x = " << pt1_cam.x - pt1_cam_3d.x << endl;
		cout << "frame_1, diff in y = " << pt1_cam.y - pt1_cam_3d.y << endl;

		cv::Point2d pt2_cam = pixel2cam(kps_2[good_matches[i].trainIdx].pt, K);
		cv::Mat pt2_cam_3d = R * (cv::Mat_<double>(3, 1) << points_3d[i].x, points_3d[i].y, points_3d[i].z) + t;
		pt2_cam_3d /= pt2_cam_3d.at<double>(2, 0);
		//cout << "point in the second camera frame: " << pt2_cam << endl;
		//cout << "point in the second camera frame projected from 3D: " << pt2_cam_3d.t() << endl << endl;
		cout << "frame_2, diff in x = " << pt2_cam.x - pt2_cam_3d.at<double>(0, 0) << endl;
		cout << "frame_2, diff in y = " << pt2_cam.y - pt2_cam_3d.at<double>(1, 0) << endl;
	}

	return 0;
}
