#include "../include/triangulation.hpp"
#include "../include/coordinate_transform.hpp"

void triangulation(const vector<cv::KeyPoint>& kps_1,
				   const vector<cv::KeyPoint>& kps_2,
				   const vector<cv::DMatch>& matches,
				   const cv::Mat& R, const cv::Mat& t,
				   vector<cv::Point3d>& points_3d){

	// R, t from kps_1 to kps_2
	cv::Mat T1 = (cv::Mat_<double> (3, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);

	cv::Mat T2 = (cv::Mat_<double> (3, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

	cv::Mat K = (cv::Mat_<double>(3, 3) << 
		520.9, 0, 325.1,
		0, 521.0, 249.7,
		0, 0, 1);
	std::vector<cv::Point2f> pts_1, pts_2;
	for(int i = 0; i < matches.size(); i++){
		pts_1.push_back( pixel2cam(kps_1[matches[i].queryIdx].pt, K) );
		pts_2.push_back( pixel2cam(kps_2[matches[i].trainIdx].pt, K) );
	}

	cv::Mat point_4d;
	cv::triangulatePoints(T1, T2, pts_1, pts_2, point_4d); // attation: pts_1 and pts_2 need to be float type

	for(int i = 0; i < point_4d.cols; i++){
		cv::Mat x = point_4d.col(i);
		x /= x.at<float>(3, 0);
		cv::Point3d p(
			x.at<float>(0, 0), 
			x.at<float>(1, 0), 
			x.at<float>(2, 0));
	
		points_3d.push_back(p);
	}
}