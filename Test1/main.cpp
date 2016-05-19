/**
 * 論文Image-based procedural modeling of facadesの2.4節に基づいて、
 * facadeを分割するテスト。
 * 分割結果は、result.pngに保存される。
 *
 * @author Gen Nishida
 * @version 1.0
 */

#include <algorithm>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#ifndef M_PI
#define M_PI	3.1415926535
#endif

using namespace std;

float gause(float u, float sigma) {
	return 1.0f / 2.0f / M_PI / sigma / sigma * expf(-u * u / 2.0f / sigma / sigma);
}

int main() {
	cv::Mat img = cv::imread("../facade/facade.png");
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

	float alpha = 0.9f;

	// compute hor(x,y) and ver(x,y) according to Equation (3)
	cv::Mat_<float> hor(grayImg.rows, grayImg.cols);
	cv::Mat_<float> ver(grayImg.rows, grayImg.cols);
	for (int r = 0; r < grayImg.rows; ++r) {
		for (int c = 0; c < grayImg.cols; ++c) {
			float dIdx;
			float dIdy;

			if (c == 0) {
				dIdx = grayImg.at<unsigned char>(r, c + 1) - grayImg.at<unsigned char>(r, c);
			}
			else if (c == grayImg.cols - 1) {
				dIdx = grayImg.at<unsigned char>(r, c) - grayImg.at<unsigned char>(r, c - 1);
			}
			else {
				dIdx = (grayImg.at<unsigned char>(r, c + 1) - grayImg.at<unsigned char>(r, c - 1)) / 2.0f;
			}

			if (r == 0) {
				dIdy = grayImg.at<unsigned char>(r + 1, c) - grayImg.at<unsigned char>(r, c);
			}
			else if (r == grayImg.rows - 1) {
				dIdy = grayImg.at<unsigned char>(r, c) - grayImg.at<unsigned char>(r - 1, c);
			}
			else {
				dIdy = (float)(grayImg.at<unsigned char>(r + 1, c) - grayImg.at<unsigned char>(r - 1, c)) / 2.0f;
			}

			hor(r, c) = std::max(dIdy * dIdy * (1.0f - alpha) - dIdx * dIdx * alpha, 0.0f);
			ver(r, c) = std::max(dIdx * dIdx * (1.0f - alpha) - dIdy * dIdy * alpha, 0.0f);
		}
	}

	// sum up the ver(x, y) horizontally and vertically, respectively
	cv::Mat ver_xtotal;
	cv::Mat ver_ytotal;
	cv::reduce(ver, ver_xtotal, 1, cv::REDUCE_SUM);
	cv::reduce(ver, ver_ytotal, 0, cv::REDUCE_SUM);

	// sum up the hor(x, y) horizontally and vertically, respectively
	cv::Mat hor_xtotal;
	cv::Mat hor_ytotal;
	cv::reduce(hor, hor_xtotal, 1, cv::REDUCE_SUM);
	cv::reduce(ver, hor_ytotal, 0, cv::REDUCE_SUM);

	// compute Ver(y) and Hor(x) according to Equation (4)
	cv::Mat_<float> Ver(grayImg.rows, 1, 0.0f);
	cv::Mat_<float> Hor(1, grayImg.cols, 0.0f);
	float sigma = 50.0f;
	float beta = 0.1f;
	for (int r = 0; r < grayImg.rows; ++r) {
		for (int rr = 0; rr < grayImg.rows; ++rr) {
			Ver(r, 0) += (ver_xtotal.at<float>(rr, 0) - beta * hor_xtotal.at<float>(rr, 0)) * gause(rr - r, sigma);
		}
	}
	for (int c = 0; c < grayImg.cols; ++c) {
		for (int cc = 0; cc < grayImg.cols; ++cc) {
			Hor(0, c) += (ver_ytotal.at<float>(0, cc) - beta * hor_ytotal.at<float>(0, cc)) * gause(cc - c, sigma);
		}
	}

	// output Ver(y) and Hor(x)
	ofstream outV("ver.txt");
	for (int r = 0; r < Ver.rows; ++r) {
		outV << Ver(r, 0) << endl;
	}
	outV.close();
	ofstream outH("hor.txt");
	for (int c = 0; c < Hor.cols; ++c) {
		outH << Hor(0, c) << endl;
	}
	outH.close();

	// find the local minima of Ver(y) and Hor(x)
	vector<int> y_set;
	for (int r = 1; r < Ver.rows - 1; ++r) {
		if (Ver(r, 0) < Ver(r - 1, 0) && Ver(r, 0) < Ver(r + 1, 0)) {
			y_set.push_back(r);
		}
	}
	vector<int> x_set;
	for (int c = 1; c < Hor.cols - 1; ++c) {
		if (Hor(0, c) < Hor(0, c - 1) && Hor(0, c) < Hor(0, c + 1)) {
			x_set.push_back(c);
		}
	}

	// visualize the Ver(y) and Hor(x)
	cv::Mat result(img.rows + 100, img.cols + 100, CV_8UC3, cv::Scalar(255, 255, 255));
	for (int rr = 0; rr < result.rows; ++rr) {
		for (int cc = 0; cc < result.cols - 1; ++cc) {
			if (rr >= img.rows && cc < img.cols - 1) {
				int rr1 = 100 - std::max(Hor(0, cc), Hor(0, cc + 1));
				int rr2 = 100 - std::min(Hor(0, cc), Hor(0, cc + 1));
				if (rr - img.rows >= rr1 && rr - img.rows <= rr2) {
					result.at<cv::Vec3b>(rr, cc) = cv::Vec3b(0, 0, 0);
				}
			}
			else if (rr < img.rows - 1 && cc >= img.cols) {
				int cc1 = std::min(Ver(rr, 0), Ver(rr + 1, 0));
				int cc2 = std::max(Ver(rr, 0), Ver(rr + 1, 0));
				if (cc - img.cols >= cc1 && cc - img.cols <= cc2) {
					result.at<cv::Vec3b>(rr, cc) = cv::Vec3b(0, 0, 0);
				}
			}
			else if (rr < img.rows && cc < img.cols) {
				result.at<cv::Vec3b>(rr, cc) = img.at<cv::Vec3b>(rr, cc);
			}
		}
	}
	for (int i = 0; i < y_set.size(); ++i) {
		cv::line(result, cv::Point(0, y_set[i]), cv::Point(img.cols - 1, y_set[i]), cv::Scalar(255, 0, 0), 3);
	}
	for (int i = 0; i < x_set.size(); ++i) {
		cv::line(result, cv::Point(x_set[i], 0), cv::Point(x_set[i], img.rows - 1), cv::Scalar(255, 0, 0), 3);
	}
	cv::imwrite("result.png", result);

	return 0;
}