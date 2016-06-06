#include "CVUtilsTest.h"
#include "CVUtils.h"
#include <opencv2/opencv.hpp>
#include <cassert>

namespace cvutils {

	void test_msd() {
		cv::Mat A1 = (cv::Mat_<unsigned char>(3, 3) << 0, 1, 4, 2, 3, 5, 8, 1, 2);
		cv::Mat A2 = (cv::Mat_<unsigned char>(3, 3) << 4, 0, 1, 2, 2, 1, 3, 4, 0);
		assert(msd(A1, A2) == 9.0);

		cv::Mat A3 = (cv::Mat_<float>(3, 3) << -1, 5, 3.2, 1.5, 0, 2, 4, 3.8, 0);
		cv::Mat A4 = (cv::Mat_<float>(3, 3) << 3, 2, 1.2, 0.5, 4, 4, 2, 0.8, 0);
		assert(msd(A3, A4) == 7.0);

		cv::Mat A5(2, 2, CV_32FC3);
		cv::Mat A6(2, 2, CV_32FC3);
		A5.at<cv::Vec3f>(0, 0) = cv::Vec3f(0, 1, 2);
		A5.at<cv::Vec3f>(0, 1) = cv::Vec3f(2, 3, 1);
		A5.at<cv::Vec3f>(1, 0) = cv::Vec3f(1, 1, 0);
		A5.at<cv::Vec3f>(1, 1) = cv::Vec3f(3, 0, 1);
		A6.at<cv::Vec3f>(0, 0) = cv::Vec3f(1, 2, 1);
		A6.at<cv::Vec3f>(0, 1) = cv::Vec3f(1, 0, 1);
		A6.at<cv::Vec3f>(1, 0) = cv::Vec3f(3, 2, 1);
		A6.at<cv::Vec3f>(1, 1) = cv::Vec3f(1, 0, 2);

		assert(msd(A5, A6) == 6.0);

		cout << "test_msd() done." << endl;
	}

}