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
		if (max(A1) != 3.0) {
			cerr << "test_max() failed at #1." << endl;
		}

		cv::Mat A2(2, 2, CV_8UC3);
		A2.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 11, 2);
		A2.at<cv::Vec3b>(0, 1) = cv::Vec3b(2, 3, 1);
		A2.at<cv::Vec3b>(1, 0) = cv::Vec3b(21, 1, 0);
		A2.at<cv::Vec3b>(1, 1) = cv::Vec3b(123, 40, 13);
		if (max(A2) != 123.0) {
			cerr << "test_max() failed at #2." << endl;
		}

		cv::Mat A3 = (cv::Mat_<float>(3, 3) << 1, 3, 9, 4.5, 9.8, 3.2, 18.5, 20.5, 1.2);
		if (max(A3) != 20.5) {
			cerr << "test_max() failed at #3." << endl;
		}		

		cv::Mat A4 = (cv::Mat_<unsigned char>(3, 3) << 21, 33, 94, 45, 98, 32, 185, 205, 12);
		if (max(A4) != 205.0) {
			cerr << "test_max() failed at #4." << endl;
		}
		
		cout << "test_max() done." << endl;
	}

	void test_min() {
		cv::Mat A1(2, 2, CV_32FC3);
		A1.at<cv::Vec3f>(0, 0) = cv::Vec3f(-5, 1, 2);
		A1.at<cv::Vec3f>(0, 1) = cv::Vec3f(2, 3, 1);
		A1.at<cv::Vec3f>(1, 0) = cv::Vec3f(1, 1, 0);
		A1.at<cv::Vec3f>(1, 1) = cv::Vec3f(3, 0, 1);
		if (min(A1) != -5.0) {
			cerr << "test_min() failed at #1." << endl;
		}


		cv::Mat A2(2, 2, CV_8UC3);
		A2.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 11, 2);
		A2.at<cv::Vec3b>(0, 1) = cv::Vec3b(2, 3, 1);
		A2.at<cv::Vec3b>(1, 0) = cv::Vec3b(21, 1, 0);
		A2.at<cv::Vec3b>(1, 1) = cv::Vec3b(123, 40, 13);
		if (min(A2) != 0.0) {
			cerr << "test_min() failed at #2." << endl;
		}

		cv::Mat A3 = (cv::Mat_<float>(3, 3) << 1, 3, 9, 4.5, 9.8, 3.2, 18.5, 20.5, 1.2);
		if (min(A3) != 1.0) {
			cerr << "test_min() failed at #3." << endl;
		}

		cv::Mat A4 = (cv::Mat_<unsigned char>(3, 3) << 21, 33, 94, 45, 98, 32, 185, 205, 12);
		if (min(A4) != 12.0) {
			cerr << "test_min() failed at #4." << endl;
		}

		cout << "test_min() done." << endl;
	}

	void test_msd() {
		cv::Mat A1 = (cv::Mat_<unsigned char>(3, 3) << 0, 1, 4, 2, 3, 5, 8, 1, 2);
		cv::Mat A2 = (cv::Mat_<unsigned char>(3, 3) << 4, 0, 1, 2, 2, 1, 3, 4, 0);
		if (msd(A1, A2) != 9.0) {
			cerr << "test_msd() failed at #1." << endl;
		}

		cv::Mat A3 = (cv::Mat_<float>(3, 3) << -1, 5, 3.2, 1.5, 0, 2, 4, 3.8, 0);
		cv::Mat A4 = (cv::Mat_<float>(3, 3) << 3, 2, 1.2, 0.5, 4, 4, 2, 0.8, 0);
		if (msd(A3, A4) != 7.0) {
			cerr << "test_msd() failed at #2." << endl;
		}

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
		if (msd(A5, A6) != 6.0) {
			cerr << "test_msd() failed at #3." << endl;
		}

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
		if (msd(A7, A8) != 6.0) {
			cerr << "test_msd() failed at #4." << endl;
		}

		cout << "test_msd() done." << endl;
	}

	void test_corr() {
		cv::Mat A1 = (cv::Mat_<unsigned char>(3, 3) << 0, 1, 4, 2, 3, 5, 5, 1, 3);
		if (corr(A1, A1) != 1.0) {
			cerr << "test_corr() failed at #1." << endl;
		}

		cv::Mat A2 = (cv::Mat_<unsigned char>(3, 3) << 1, 1, 2, 2, 2, 3, 4, 1, 0);
		if (abs(corr(A1, A2) * 12.0 - 11.0) > 0.000001) {
			cerr << "test_corr() failed at #2." << endl;
		}

		cout << "test_corr() done." << endl;
	}
}