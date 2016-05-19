/**
* 論文Image-based procedural modeling of facadesの2.3節に基づいて、
* Irreducible facadeを計算する。
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

/**
 * y_initialの周辺で、最適なsplit位置、yを探す。 
 * その時のsimilarityを返却する。
 */
float findAdjacentSplitUpward(const cv::Mat_<float>& S_max, const cv::Mat_<int>& h_max, int y_initial, int h, int& y) {
	y = y_initial;
	float S = S_max(y, 0);

	for (int r = std::max(y_initial - 3, 0); r <= y_initial + 3; ++r) {
		if (abs(h_max(r, 0) - h) > 3) continue;

		if (S_max(r, 0) > S) {
			y = r;
			S = S_max(r, 0);
		}
	}

	return S;
}

/**
 * y_startから開始し、上方向へ、最適なsplit位置、next_yを探す。
 * その時のsimilarityを返却する。
 */
float findNextSplitUpward(const cv::Mat_<float>& S_max, const cv::Mat_<int>& h_max, int y_start, int& next_y) {
	float S = 0.0f;

	for (int r = y_start; r >= 0; --r) {
		if (r - h_max(r, 0) < 0) continue;
		if (abs(r + h_max(r, 0) - y_start) > 3) continue;

		if (S_max(r, 0) > S) {
			next_y = r;
			S = S_max(r, 0);
		}
	}

	return S;
}

void findSplitsUpward(const cv::Mat_<float>& S_max, const cv::Mat_<int>& h_max, int y, int h, float tau_max, vector<int>& y_set) {
	while (true) {
		int next_y;
		float S = findAdjacentSplitUpward(S_max, h_max, y, h, next_y);

		cout << "y: " << next_y << ", S: " << S << ", h: " << h_max(next_y, 0) << endl;

		if (S >= tau_max * 0.75f) {
			y_set.push_back(next_y);
			y = next_y - h;
		}
		else {
			cout << " --> not good" << endl;

			S = findNextSplitUpward(S_max, h_max, y, next_y);
			if (S >= tau_max * 0.75f) {
				y_set.push_back(next_y);
				y = next_y - h_max(next_y, 0);
				h = h_max(next_y, 0);
				cout << "  --> modified. y: " << next_y << ", S: " << S << ", h: " << h_max(next_y, 0) << endl;
			}
			else {
				return;
			}
		}
	}
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

	// find the maximum of S_max(y)
	float S = 0.0f;
	int y;
	for (int r = 0; r < S_max.rows; ++r) {
		if (S_max(r, 0) > S) {
			y = r;
			S = S_max(r, 0);
		}
	}

	vector<int> y_set;
	y_set.push_back(y);

	cout << "y: " << y << ", S: " << S << ", h: " << h_max(y, 0) << endl;

	// check upward
	findSplitsUpward(S_max, h_max, y - h_max(y, 0), h_max(y, 0), S, y_set);
	
	cout << "Terminated." << endl;

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
	for (int i = 0; i < y_set.size(); ++i) {
		cout << "Y: " << y_set[i] << ", h: " << h_max(y_set[i], 0) << endl;

		cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
		cv::rectangle(result, cv::Rect(0, y_set[i] - h_max(y_set[i], 0), img.cols, h_max(y_set[i], 0)), color, 3);
		cv::rectangle(result, cv::Rect(0, y_set[i], img.cols, h_max(y_set[i], 0)), color, 3);
	}
	cv::imwrite("result.png", result);


	return 0;
}