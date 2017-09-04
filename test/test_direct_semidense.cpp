#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <Eigen/Dense>

#include "../include/coordinate_transform.hpp"
#include "../include/direct_sparse_g2o.hpp"

using namespace std;

/********************************************
 * 				RGBD上的半稠密直接法
 ********************************************/

int main(int argc, char** argv){
	if(argc != 2){
		cout << "usage: useLK path_to_dataset" << endl;
		return 1;
	}
	srand((unsigned int) time(0));
	string path_to_dataset = argv[1];
	string associate_file = path_to_dataset + "/associate.txt";

	ifstream fin(associate_file);

	string rgb_file, depth_file, time_rgb, time_depth;
	cv::Mat color, prev_color, depth, gray;
	std::vector<Measurement> measurements;

	Eigen::Matrix3d K;
	K << 518.0, 0, 325.5,
		 0, 519.0, 253.5,
		 0, 0, 1;
	double depth_scale = 1000.0;

	Eigen::Isometry3d T_cw = Eigen::Isometry3d::Identity();

	cv::namedWindow("result");
	cv::Mat img_show;

	// 我们以第一个图像为参考，对后续图像和参考图像做直接法
	for(int index = 0; index < 10; index++){
		cout << "************* loop " << index << " **************" << endl;
		fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
		color = cv::imread(path_to_dataset + "/" +rgb_file);
		depth = cv::imread(path_to_dataset + "/" + depth_file, -1);
		if(color.data == NULL || depth.data == NULL)
			continue;
		cv::cvtColor ( color, gray, cv::COLOR_BGR2GRAY );

		if(index == 0){
			
			// select the pixels with high gradiants
			for(int x = 10; x < gray.cols - 10; x++)
				for(int y = 10; y < gray.rows - 10; y++){
					Eigen::Vector2d delta(
						gray.ptr<uchar>(y)[x + 1] - gray.ptr<uchar>(y)[x - 1],
						gray.ptr<uchar>(y + 1)[x] - gray.ptr<uchar>(y - 1)[x]);
					if(delta.norm() < 50)
						continue;
					ushort d = depth.ptr<ushort>(y)[x];
					if(d == 0)
						continue;
					Eigen::Vector3d p3d = project2Dto3D(x, y, d, K, depth_scale);
					double grayscale = (double) gray.ptr<uchar>(y)[x];
					measurements.push_back( Measurement(p3d, grayscale) );
				}
			prev_color = color.clone();

			img_show.create(color.rows, color.cols * 2, CV_8UC3);
			prev_color.copyTo ( img_show ( cv::Rect ( 0, 0, color.cols, color.rows ) ) );
			continue;
		}

		// 使用直接法计算相机运动
		clock_t t1 = clock();
		direct_sparse_g2o(measurements, gray, K, T_cw);
		clock_t t2 = clock();
		cout << "g2o direct sparse costs time: " << 1.0 * (t2 - t1) / CLOCKS_PER_SEC << " seconds.\n";
		cout << "T_cw = \n" << T_cw.matrix() << endl;


		// plot the feature points    
        color.copyTo ( img_show ( cv::Rect ( color.cols, 0, color.cols, color.rows ) ) );
        for ( Measurement m:measurements )
        {
            if ( rand() > RAND_MAX/5 )
                continue;
            Eigen::Vector3d p = m._p_world;
            Eigen::Vector2d pixel_prev = project3Dto2D ( p[0], p[1], p[2], K);
            Eigen::Vector3d p2 = T_cw * p;
            Eigen::Vector2d pixel_now = project3Dto2D ( p2[0], p2[1], p2[2], K);
            if ( pixel_now(0,0)<3 || pixel_now(0,0)+3>=color.cols || pixel_now(1,0)<3 || pixel_now(1,0)+3>=color.rows )
                continue;

            float b = 255*float ( rand() ) /RAND_MAX;
            float g = 255*float ( rand() ) /RAND_MAX;
            float r = 255*float ( rand() ) /RAND_MAX;
            cv::circle ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 8, cv::Scalar ( b,g,r ), 2 );
            cv::circle ( img_show, cv::Point2d ( pixel_now ( 0,0 ) + color.cols, pixel_now ( 1,0 ) ), 8, cv::Scalar ( b,g,r ), 2 );
            //cv::line ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +color.rows ), cv::Scalar ( b,g,r ), 1 );
        }
        cv::imshow ( "result", img_show );
        cv::waitKey ( 0 );
	}
	cv::destroyWindow("result");
	return 0;
}