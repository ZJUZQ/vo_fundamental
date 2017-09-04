#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <time.h>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>

int main(int argc, char** argv){
	if(argc != 2){
		cout << "usage: ./useLK path_to_dataset" << endl;
		return 1;
	}
	string path_to_dataset = argv[1];
	string associate_file = path_to_dataset + "/associate.txt";
	ifstream fin(associate_file.c_str());

	string rgb_file, depth_file, time_rgb, time_depth;
	std::list<cv::Point2f> kps_track; // // 因为要删除跟踪失败的点，使用list
	cv::Mat color, depth, last_color;

	for(int index = 0; index < 100; index++){
		fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
		color = cv::imread(path_to_dataset + "/" + rgb_file);
		depth = cv::imread(path_to_dataset + "/" + depth_file, -1);
		if(index == 0){
			// extract FAST corner in the first frame
			std::vector<cv::KeyPoint> kps;
			cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
			detector->detect(color, kps);
			for(int i = 0; i < kps.size(); i++)
				kps_track.push_back(kps[i].pt);
			last_color = color.clone();
			continue;
		}
		if(color.data == NULL || depth.data == NULL)
			continue;

		//tracking with LK
		vector<cv::Point2f> next_kps;
		vector<cv::Point2f> pre_kps;
		for(list<cv::Point2f>::iterator iter = kps_track.begin(); iter != kps_track.end(); iter++)
			pre_kps.push_back(*iter);
		std::vector<unsigned char> status;
		std::vector<float> errs;
		clock_t t1 = clock();

		// point coordinates must be single-precision floating-point numbers
		cv::calcOpticalFlowPyrLK(last_color, color, pre_kps, next_kps, status, errs);

		clock_t t2 = clock();
		cout << "LK use time: " << (t2 - t1) * 1000.0 / CLOCKS_PER_SEC << " ms" << endl;
		cout << "pre_kps.size() = " << pre_kps.size() << endl;
		cout << "next_kps.size() = " << next_kps.size() << endl;

		// delete lost keypoints, and update the track_points' position
		int i = 0;
		for(list<cv::Point2f>::iterator iter = kps_track.begin(); iter != kps_track.end(); i++){
			if(status[i] == 0){
				iter = kps_track.erase(iter);
				continue;
			}
			*iter = next_kps[i];
			iter++;
		}
		cout << "tracked keyponts: " << kps_track.size() << endl;
		if(kps_track.size() == 0){
			cout << "all keyponts lost."  << endl;
			break;
		}

		cv::Mat img = color.clone();
		for(list<cv::Point2f>::iterator iter = kps_track.begin(); iter != kps_track.end(); iter++)
			cv::circle(img, *iter, 6, cv::Scalar(0, 255, 0), 1);
		cv::imshow("LK", img);
		cv::waitKey(0);

		last_color = color.clone();
	}
	return 0;
}