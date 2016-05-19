/**
* 論文Image-based procedural modeling of facadesの2.1、2.2節に基づいて、
* S_max(y)とh_max(y)を計算するテスト。
* 結果は、result.pngに保存される。
*
* @author Gen Nishida
* @version 1.0
*/

#include <algorithm>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;

float MI(const cv::Mat& R1, const cv::Mat& R2) {
	cv::Mat_<float> Pab(256, 256, 0.0f);
	cv::Mat_<float> Pa(256, 1, 0.0f);
	cv::Mat_<float> Pb(256, 1, 0.0f);

	// create a histogram of intensities
	for (int r = 0; r < R1.rows; ++r) {
		for (int c = 0; c < R1.cols; ++c) {
			int a = R1.at<unsigned char>(r, c);
			int b = R2.at<unsigned char>(r, c);

			Pab(a, b)++;
			Pa(a, 0)++;
			Pb(b, 0)++;
		}
	}

	// normalize the historgram
	cv::Mat Pab_max;
	cv::Mat Pa_max;
	cv::Mat Pb_max;
	cv::reduce(Pab, Pab_max, 0, CV_REDUCE_SUM);
	cv::reduce(Pab_max, Pab_max, 1, CV_REDUCE_SUM);
	cv::reduce(Pa, Pa_max, 0, CV_REDUCE_SUM);
	cv::reduce(Pb, Pb_max, 0, CV_REDUCE_SUM);
	Pab /= Pab_max.at<float>(0, 0);
	Pa /= Pa_max.at<float>(0, 0);
	Pb /= Pb_max.at<float>(0, 0);

	float result = 0.0f;
	for (int a = 0; a < 256; ++a) {
		for (int b = 0; b < 256; ++b) {
			float v = Pab(a, b);
			if (v == 0) {
				v = 0.001f * 0.001f;
			}
			float v1 = Pa(a, 0);
			if (v1 == 0) v1 = 0.001f;
			float v2 = Pb(b, 0);
			if (v2 == 0) v2 = 0.001f;

			float hoge = Pab(a, b) * log(v / v1 / v2);
			result += Pab(a, b) * log(v / v1 / v2);
		}
	}

	return result;
}

int main() {
	cv::Mat img = cv::imread("../facade/facade.png");
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

	cv::Mat_<float> S_max(img.rows, 1, 0.0f);
	cv::Mat_<int> h_max(img.rows, 1, 0.0f);

	printf("computing");
	for (int r = 0; r < grayImg.rows; ++r) {
		printf("\rcomputing r = %d/%d  ", r, grayImg.rows);

		for (int h = 80; h < 200; ++h) {
			if (r - h < 0 || r + h >= grayImg.rows) continue;

			cv::Mat R1 = grayImg(cv::Rect(0, r, grayImg.cols, h));
			cv::Mat R2 = grayImg(cv::Rect(0, r - h, grayImg.cols, h));
			float S = MI(R1, R2);
			if (S > S_max(r, 0)) {
				S_max(r, 0) = S;
				h_max(r, 0) = h;
			}
		}
	}
	printf("\n");

	// output S_max(y) and h_max(y)
	ofstream outS("S_max.txt");
	for (int r = 0; r < S_max.rows; ++r) {
		outS << S_max(r, 0) << endl;
	}
	outS.close();
	ofstream outh("h_max.txt");
	for (int r = 0; r < h_max.rows; ++r) {
		outh << h_max(r, 0) << endl;
	}
	outh.close();

	// normalize S_max(y) to the range [0, 1]
	cv::Mat S_max_max;
	cv::reduce(S_max, S_max_max, 0, cv::REDUCE_MAX);
	S_max /= S_max_max.at<float>(0, 0);

	// store the S_max and h_max in the vector
	vector<pair<int, pair<float, int>>> y_set;
	for (int i = 0; i < S_max.rows; ++i) {
		y_set.push_back(make_pair(i, make_pair(S_max(i, 0), h_max(i, 0))));
	}

	// find the maximum of S_max(y)
	int y1;
	int h1;
	for (int r = 0; r < S_max.rows; ++r) {
		if (S_max(r, 0) == 1.0f) {
			y1 = r;
			h1 = h_max(r, 0);

			// remove the region R1 and R2
			for (int i = y_set.size() - 1; i >= 0; --i) {
				if (y_set[i].first >= y1 - h1 && y_set[i].first < y1 + h1) {
					y_set.erase(y_set.begin() + i);
				}
			}

			break;
		}
	}

	// find the second maximum of S_max(y)
	float S2 = 0.0f;
	int y2;
	int h2;
	for (int i = 0; i < y_set.size(); ++i) {
		if (y_set[i].second.first > S2) {
			y2 = y_set[i].first;
			S2 = y_set[i].second.first;
			h2 = y_set[i].second.second;
		}
	}

	// visualize S_max(y) and h_max(y)
	cv::Mat result(img.rows, img.cols + 200 + 300, CV_8UC3, cv::Scalar(255, 255, 255));
	for (int r = 0; r < img.rows; ++r) {
		for (int c = 0; c < img.cols + 200 + 300; ++c) {
			if (c < img.cols) {
				result.at<cv::Vec3b>(r, c) = img.at<cv::Vec3b>(r, c);
			}
			else if (c < img.cols + 200 && r < img.rows - 1) {
				int v1 = std::min(S_max(r, 0), S_max(r + 1, 0)) * 200;
				int v2 = std::max(S_max(r, 0), S_max(r + 1, 0)) * 200;
				if (c - img.cols >= v1 && c - img.cols <= v2) {
					result.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
				}
			}
			else if (c >= img.cols + 200 && r < img.rows - 1) {
				int v1 = std::min(h_max(r, 0), h_max(r + 1, 0));
				int v2 = std::max(h_max(r, 0), h_max(r + 1, 0));
				if (c - img.cols - 200 >= v1 && c - img.cols - 200 <= v2) {
					result.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
				}
			}
		}
	}
	cv::rectangle(result, cv::Rect(0, y1 - h1, img.cols, h1), cv::Scalar(255, 0, 0), 3);
	cv::rectangle(result, cv::Rect(0, y1, img.cols, h1), cv::Scalar(255, 0, 0), 3);
	cv::rectangle(result, cv::Rect(0, y2 - h2, img.cols, h2), cv::Scalar(0, 0, 255), 3);
	cv::rectangle(result, cv::Rect(0, y2, img.cols, h2), cv::Scalar(0, 0, 255), 3);
	cv::imwrite("result.png", result);


	return 0;
}