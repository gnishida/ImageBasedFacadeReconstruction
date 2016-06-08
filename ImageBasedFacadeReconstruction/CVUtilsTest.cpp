#include "CVUtilsTest.h"
#include "CVUtils.h"
#include <opencv2/opencv.hpp>
#include <cassert>

namespace cvutils {

	void test_cvutils() {
		test_max();
		test_min();
		test_msd();
		test_corr();
	}

	void test_max() {
		cv::Mat A1(2, 2, CV_32FC3);
		A1.at<cv::Vec3f>(0, 0) = cv::Vec3f(0, 1, 2);
		A1.at<cv::Vec3f>(0, 1) = cv::Vec3f(2, 3, 1);
		A1.at<cv::Vec3f>(1, 0) = cv::Vec3f(1, 1, 0);
		A1.at<cv::Vec3f>(1, 1) = cv::Vec3f(3, 0, 1);
		assert(max(A1) == 3.0);

		cv::Mat A2(2, 2, CV_8UC3);
		A2.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 11, 2);
		A2.at<cv::Vec3b>(0, 1) = cv::Vec3b(2, 3, 1);
		A2.at<cv::Vec3b>(1, 0) = cv::Vec3b(21, 1, 0);
		A2.at<cv::Vec3b>(1, 1) = cv::Vec3b(123, 40, 13);
		assert(max(A2) == 123.0);

		cv::Mat A3 = (cv::Mat_<float>(3, 3) << 1, 3, 9, 4.5, 9.8, 3.2, 18.5, 20.5, 1.2);
		assert(max(A3) == 20.5);

		cv::Mat A4 = (cv::Mat_<unsigned char>(3, 3) << 21, 33, 94, 45, 98, 32, 185, 205, 12);
		assert(max(A4) == 205.0);

		cout << "test_max() done." << endl;
	}

	void test_min() {
		cv::Mat A1(2, 2, CV_32FC3);
		A1.at<cv::Vec3f>(0, 0) = cv::Vec3f(-5, 1, 2);
		A1.at<cv::Vec3f>(0, 1) = cv::Vec3f(2, 3, 1);
		A1.at<cv::Vec3f>(1, 0) = cv::Vec3f(1, 1, 0);
		A1.at<cv::Vec3f>(1, 1) = cv::Vec3f(3, 0, 1);
		assert(min(A1) == -5.0);

		cv::Mat A2(2, 2, CV_8UC3);
		A2.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 11, 2);
		A2.at<cv::Vec3b>(0, 1) = cv::Vec3b(2, 3, 1);
		A2.at<cv::Vec3b>(1, 0) = cv::Vec3b(21, 1, 0);
		A2.at<cv::Vec3b>(1, 1) = cv::Vec3b(123, 40, 13);
		assert(min(A2) == 0.0);

		cv::Mat A3 = (cv::Mat_<float>(3, 3) << 1, 3, 9, 4.5, 9.8, 3.2, 18.5, 20.5, 1.2);
		assert(min(A3) == 1.0);

		cv::Mat A4 = (cv::Mat_<unsigned char>(3, 3) << 21, 33, 94, 45, 98, 32, 185, 205, 12);
		assert(min(A4) == 12.0);

		cout << "test_min() done." << endl;
	}

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

		cv::Mat A7(2, 2, CV_8UC3);
		cv::Mat A8(2, 2, CV_8UC3);
		A7.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 1, 2);
		A7.at<cv::Vec3b>(0, 1) = cv::Vec3b(2, 3, 1);
		A7.at<cv::Vec3b>(1, 0) = cv::Vec3b(1, 1, 0);
		A7.at<cv::Vec3b>(1, 1) = cv::Vec3b(3, 0, 1);
		A8.at<cv::Vec3b>(0, 0) = cv::Vec3b(1, 2, 1);
		A8.at<cv::Vec3b>(0, 1) = cv::Vec3b(1, 0, 1);
		A8.at<cv::Vec3b>(1, 0) = cv::Vec3b(3, 2, 1);
		A8.at<cv::Vec3b>(1, 1) = cv::Vec3b(1, 0, 2);
		assert(msd(A7, A8) == 6.0);

		cout << "test_msd() done." << endl;
	}

	void test_corr() {
		cv::Mat A1 = (cv::Mat_<unsigned char>(3, 3) << 0, 1, 4, 2, 3, 5, 5, 1, 3);
		if (corr(A1, A1) != 1.0) {
			cerr << "test_corr() failed #1." << endl;
		}

		cv::Mat A2 = (cv::Mat_<unsigned char>(3, 3) << 1, 1, 2, 2, 2, 3, 4, 1, 0);
		if (abs(corr(A1, A2) * 12.0 - 11.0) > 0.000001) {
			cerr << "test_corr() failed #2." << endl;
		}

		cout << "test_corr() done." << endl;
	}
}