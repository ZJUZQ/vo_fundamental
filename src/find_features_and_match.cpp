#include "../include/find_features_and_match.hpp"

void find_features_and_match(cv::Mat &img_1, 
						       cv::Mat &img_2,
						  	   std::vector<cv::KeyPoint> &kps_1,
						  	   std::vector<cv::KeyPoint> &kps_2,
						  	   std::vector<cv::DMatch> &good_matches){

	cv::Mat descriptors_1, descriptors_2;
	/**
		static Ptr<ORB> cv::ORB::create 	( 	int  	nfeatures = 500,
												float  	scaleFactor = 1.2f,
												int  	nlevels = 8,
												int  	edgeThreshold = 31,
												int  	firstLevel = 0,
												int  	WTA_K = 2,
												int  	scoreType = ORB::HARRIS_SCORE,
												int  	patchSize = 31,
												int  	fastThreshold = 20 ) 	
	*/
	cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

	orb->detect(img_1, kps_1);
	orb->compute(img_1, kps_1, descriptors_1);
	orb->detect(img_2, kps_2);
	orb->compute(img_2, kps_2, descriptors_2);

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

	for(int i = 0; i < descriptors_1.rows; i++){
		if(matches[i].distance <= 30.0 || matches[i].distance <= 2 * min_dist)
			good_matches.push_back(matches[i]);
	}
}