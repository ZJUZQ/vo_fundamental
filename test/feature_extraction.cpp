#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

int main(int argc, char** argv){
	if(argc != 3){
		cout << "usage: ./feature_extraction img1 img2" << endl;
		return 1;
	}
	cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);

	vector<cv::KeyPoint> kps_1, kps_2;
	cv::Mat descriptors_1, descriptors_2;
	cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

	orb->detect(img_1, kps_1);
	orb->compute(img_1, kps_1, descriptors_1);
	orb->detect(img_2, kps_2);
	orb->compute(img_2, kps_2, descriptors_2);

	cv::Mat out_img_1;
	cv::drawKeypoints(img_1, kps_1, out_img_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
	cv::imshow("img1_kps", out_img_1);

	std::vector<cv::DMatch> matches;
	cv::BFMatcher matcher(cv::NORM_HAMMING);
	matcher.match(descriptors_1, descriptors_2, matches);

	double min_dist = 10000, max_dist = 0;
	for(int i = 0; i < kps_1.size(); i++){
		double dist = matches[i].distance;
		min_dist = dist < min_dist ? dist : min_dist;
		max_dist = dist > max_dist ? dist : max_dist;
	}
	printf("--Max distance is: %.2f \n", max_dist);
	printf("--Min distance is: %.2f \n", min_dist);

	std::vector<cv::DMatch> good_matches;
	for(int i = 0; i < descriptors_1.rows; i++){i		if(matches[i].distance <= 30.0 || matches[i].distance <= 2 * min_dist)
			good_matches.push_back(matches[i]);
	}

	cv::Mat img_matches, img_goodmatches;
	cv::drawMatches(img_1, kps_1, img_2, kps_2, matches, img_matches);
	cv::drawMatches(img_1, kps_1, img_2, kps_2, good_matches, img_goodmatches);
	cv::imshow("all_matches", img_matches);
	cv::imshow("good_matches", img_goodmatches);
	cv::waitKey(0);

	cv::destroyAllWindows();
	return 0;
}