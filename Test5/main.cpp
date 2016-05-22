/**
* 論文Image-based procedural modeling of facadesの3章に基づいて、
* Facadeの各tileをさらに分割する。
*
* @author Gen Nishida
* @version 1.0
*/

#include <algorithm>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "CVUtils.h"
#include "Utils.h"

using namespace std;

void outputFacadeStructureV(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& h_max, const vector<int>& y_set, const string& filename) {
	float S_max_max = cvutils::max(S_max);

	cv::Mat result(img.rows, img.cols + 200 + 300, CV_8UC3, cv::Scalar(255, 255, 255));
	for (int r = 0; r < img.rows; ++r) {
		for (int c = 0; c < img.cols + 200 + 300; ++c) {
			if (c < img.cols) {
				result.at<cv::Vec3b>(r, c) = img.at<cv::Vec3b>(r, c);
			}
			else if (c < img.cols + 200 && r < img.rows - 1) {
				int v1 = std::min(S_max(r, 0), S_max(r + 1, 0)) / S_max_max * 200;
				int v2 = std::max(S_max(r, 0), S_max(r + 1, 0)) / S_max_max * 200;
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
	cv::imwrite(filename, result);
}

void outputFacadeStructureH(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& w_max, const vector<int>& x_set, const string& filename) {
	float S_max_max = cvutils::max(S_max);

	cv::Mat result(img.rows + 200 + 300, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	for (int r = 0; r < img.rows + 200 + 300; ++r) {
		for (int c = 0; c < img.cols; ++c) {
			if (r < img.rows) {
				result.at<cv::Vec3b>(r, c) = img.at<cv::Vec3b>(r, c);
			}
			else if (r < img.rows + 200 && c < img.cols - 1) {
				int v1 = std::min(S_max(0, c), S_max(0, c + 1)) / S_max_max * 200;
				int v2 = std::max(S_max(0, c), S_max(0, c + 1)) / S_max_max * 200;
				if (r - img.rows >= v1 && r - img.rows <= v2) {
					result.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
				}
			}
			else if (r >= img.rows + 200 && c < img.cols - 1) {
				int v1 = std::min(w_max(0, c), w_max(0, c + 1));
				int v2 = std::max(w_max(0, c), w_max(0, c + 1));
				if (r - img.rows - 200 >= v1 && r - img.rows - 200 <= v2) {
					result.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
				}
			}
		}
	}
	for (int i = 0; i < x_set.size(); ++i) {
		cout << "X: " << x_set[i] << ", w: " << w_max(0, x_set[i]) << endl;

		cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
		cv::rectangle(result, cv::Rect(x_set[i] - w_max(0, x_set[i]), 0, w_max(0, x_set[i]), img.rows), color, 3);
		cv::rectangle(result, cv::Rect(x_set[i], 0, w_max(0, x_set[i]), img.rows), color, 3);
	}
	cv::imwrite(filename, result);
}

void outputFacadeStructure(const cv::Mat& img, const vector<int>& y_set, const vector<int>& x_set, const string& filename) {
	cv::Mat result = img.clone();


	for (int i = 0; i < y_set.size(); ++i) {
		cv::line(result, cv::Point(0, y_set[i]), cv::Point(img.cols, y_set[i]), cv::Scalar(0, 0, 255), 3);
	}
	for (int i = 0; i < x_set.size(); ++i) {
		cv::line(result, cv::Point(x_set[i], 0), cv::Point(x_set[i], img.rows), cv::Scalar(0, 0, 255), 3);
	}
	cv::imwrite(filename, result);
}

/**
* IFデータに基づいて画像を生成する。
*
* @param	IF		IrreducibleFacade
* @param	imgIF	Image of IF
*/
void createIFImage(const cv::Mat& IF, cv::Mat& imgIF) {
	imgIF = cv::Mat(IF.rows, IF.cols, CV_8UC3);

	for (int r = 0; r < IF.rows; ++r) {
		for (int c = 0; c < IF.cols; ++c) {
			int blue = IF.at<cv::Vec4f>(r, c)[0] / IF.at<cv::Vec4f>(r, c)[3];
			int green = IF.at<cv::Vec4f>(r, c)[1] / IF.at<cv::Vec4f>(r, c)[3];
			int red = IF.at<cv::Vec4f>(r, c)[2] / IF.at<cv::Vec4f>(r, c)[3];
			imgIF.at<cv::Vec3b>(r, c) = cv::Vec3b(blue, green, red);
		}
	}
}

void outputIF(const cv::Mat& IF, const string& filename) {
	cv::Mat imgIF;
	createIFImage(IF, imgIF);

	cv::imwrite(filename, imgIF);
}

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
* Irreducible Facadeの、[y, y+h]を[y-h, y]へ移動してshrinkさせる。
*
* @param	IF		Irreducible Facade
* @param	y		Splitの位置
* @param	h		Overlapさせる高さ
*/
void vshrinkIF(cv::Mat& IF, int y, int h) {
	h = std::min(h, IF.rows - y);

	cv::Mat IF_bak = IF.clone();
	IF = cv::Mat(IF.rows - h, IF.cols, CV_32FC4);

	// overlapping部分より上側をコピー
	for (int r = 0; r < y; ++r) {
		for (int c = 0; c < IF.cols; ++c) {
			IF.at<cv::Vec4f>(r, c) = IF_bak.at<cv::Vec4f>(r, c);
		}
	}

	// overlapping部分をコピー
	for (int r = y - h; r < y; ++r) {
		for (int c = 0; c < IF.cols; ++c) {
			IF.at<cv::Vec4f>(r, c) = IF.at<cv::Vec4f>(r, c) + IF_bak.at<cv::Vec4f>(r + h, c);
		}
	}

	// overlapping部分より下側をコピー
	for (int r = y; r < IF.rows; ++r) {
		for (int c = 0; c < IF.cols; ++c) {
			IF.at<cv::Vec4f>(r, c) = IF_bak.at<cv::Vec4f>(r + h, c);
		}
	}
}

/**
 * Irredicible facadeを垂直方向に、split位置y_setに従ってshrinkさせる。
 * ただし、y_setは昇順に並んでいるものとする。
 */
void vshrinkIF(cv::Mat& IF, const vector<int>& y_set, const cv::Mat_<float>& h_max) {
	for (int i = y_set.size() - 1; i >= 0; --i) {
		vshrinkIF(IF, y_set[i], h_max(y_set[i], 0));
	}
}

/**
* Irreducible Facadeの、[x, x+w]を[x-w, x]へ移動してshrinkさせる。
*
* @param	IF		Irreducible Facade
* @param	x		Splitの位置
* @param	w		Overlapさせる幅
*/
void hshrinkIF(cv::Mat& IF, int x, int w) {
	w = std::min(w, IF.cols - x);

	cv::Mat IF_bak = IF.clone();
	IF = cv::Mat(IF.rows, IF.cols - w, CV_32FC4);

	// overlapping部分より左側をコピー
	for (int c = 0; c < x; ++c) {
		for (int r = 0; r < IF.rows; ++r) {
			IF.at<cv::Vec4f>(r, c) = IF_bak.at<cv::Vec4f>(r, c);
		}
	}

	// overlapping部分をコピー
	for (int c = x - w; c < x; ++c) {
		for (int r = 0; r < IF.rows; ++r) {
			IF.at<cv::Vec4f>(r, c) = IF.at<cv::Vec4f>(r, c) + IF_bak.at<cv::Vec4f>(r, c + w);
		}
	}

	// overlapping部分より右側をコピー
	for (int c = x; c < IF.cols; ++c) {
		for (int r = 0; r < IF.rows; ++r) {
			IF.at<cv::Vec4f>(r, c) = IF_bak.at<cv::Vec4f>(r, c + w);
		}
	}
}

/**
 * Irreducible facadeを、splitラインの位置x_setに基づいてshrinkさせる。
 * ただし、x_setは、昇順に並んでいるものとする。
 */
void hshrinkIF(cv::Mat& IF, const vector<int>& x_set, const cv::Mat_<float>& w_max) {
	for (int i = x_set.size() - 1; i >= 0; --i) {
		hshrinkIF(IF, x_set[i], w_max(0, x_set[i]));
	}
}

/**
* y_initialの周辺で、最適なsplit位置、yを探す。
* その時のsimilarityを返却する。
*/
float findAdjacentVerticalSplit(const cv::Mat_<float>& S_max, const cv::Mat_<float>& h_max, int y_initial, int h, int& y) {
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
* yから開始し、上方向へ、最適なsplit位置、next_yを探す。
* その時のsimilarityを返却する。
*/
float findNextVerticalSplit(const cv::Mat_<float>& S_max, const cv::Mat_<float>& h_max, int y, int dir, int& next_y) {
	float S = 0.0f;

	for (int r = y; r >= 0 && r < S_max.rows; r += dir) {
		if (r + h_max(r, 0) * dir < 0 || r + h_max(r, 0) * dir >= S_max.rows) continue;
		if (abs(r - h_max(r, 0) * dir - y) > 3) continue;

		if (S_max(r, 0) > S) {
			next_y = r;
			S = S_max(r, 0);
		}
	}

	return S;
}

void findVerticalSplits(const cv::Mat_<float>& S_max, const cv::Mat_<float>& h_max, int y, int h, float tau_max, int dir, vector<int>& y_set) {
	while (true) {
		int next_y;
		float S = findAdjacentVerticalSplit(S_max, h_max, y, h, next_y);

		cout << "y: " << next_y << ", S: " << S << ", h: " << h_max(next_y, 0) << endl;

		if (S >= tau_max * 0.75f) {
			y_set.push_back(next_y);
			y = next_y + h * dir;
		}
		else {
			cout << " --> not good" << endl;

			S = findNextVerticalSplit(S_max, h_max, y, dir, next_y);
			if (S >= tau_max * 0.75f) {
				y_set.push_back(next_y);
				y = next_y + h_max(next_y, 0) * dir;
			}
			else {
				break;
			}
		}
	}

	cout << "Terminated." << endl;
}

/**
* x_initialの周辺で、最適なsplit位置、xを探す。
* その時のsimilarityを返却する。
*/
float findAdjacentHorizontalSplit(const cv::Mat_<float>& S_max, const cv::Mat_<float>& w_max, int x_initial, int w, int& x) {
	x = x_initial;
	float S = S_max(0, x);

	for (int c = std::max(x_initial - 3, 0); c <= x_initial + 3; ++c) {
		if (abs(w_max(0, c) - w) > 3) continue;

		if (S_max(0, c) > S) {
			x = c;
			S = S_max(0, c);
		}
	}

	return S;
}

/**
* x_startからdir方向へ探索し、次の最適なsplit位置、next_xを探す。
* その時のsimilarityを返却する。
*/
float findNextHorizontalSplit(const cv::Mat_<float>& S_max, const cv::Mat_<float>& w_max, int x, int dir, int& next_x) {
	float S = 0.0f;

	for (int c = x; c >= 0 && c < S_max.cols; c += dir) {
		if (c + w_max(0, c) * dir < 0 || c + w_max(0, c) * dir >= S_max.cols) continue;
		if (abs(c - w_max(0, c) * dir - x) > 3) continue;

		if (S_max(0, c) > S) {
			next_x = c;
			S = S_max(0, c);
		}
	}

	return S;
}

void findHorizontalSplits(const cv::Mat_<float>& S_max, const cv::Mat_<float>& w_max, int x, int w, float tau_max, int dir, vector<int>& x_set) {
	while (true) {
		int next_x;
		float S = findAdjacentHorizontalSplit(S_max, w_max, x, w, next_x);

		cout << "x: " << next_x << ", S: " << S << ", w: " << w_max(0, next_x) << endl;

		if (S >= tau_max * 0.75f) {
			x_set.push_back(next_x);
			x = next_x + w * dir;
		}
		else {
			cout << " --> not good" << endl;

			S = findNextHorizontalSplit(S_max, w_max, x, dir, next_x);
			if (S >= tau_max * 0.75f) {
				x_set.push_back(next_x);
				x = next_x + w_max(next_x, 0) * dir;
			}
			else {
				break;
			}
		}
	}

	cout << "Terminated." << endl;
}

void verticalSplit(const cv::Mat& img, vector<int>& y_set, cv::Mat& IF) {
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

	cv::Mat_<float> SV_max(img.rows, 1, 0.0f);
	cv::Mat_<float> h_max(img.rows, 1, 0.0f);

	ifstream in_SV("SV_max.txt");
	ifstream in_h("h_max.txt");
	if (in_SV.good() && in_h.good()) {
		in_SV.close();
		in_h.close();
		SV_max = cvutils::read("SV_max.txt");
		h_max = cvutils::read("h_max.txt");
	}
	else {
		printf("computing");
		for (int r = 0; r < grayImg.rows; ++r) {
			printf("\rcomputing r = %d/%d  ", r, grayImg.rows);

			for (int h = 80; h < 200; ++h) {
				if (r - h < 0 || r + h >= grayImg.rows) continue;

				cv::Mat R1 = grayImg(cv::Rect(0, r, grayImg.cols, h));
				cv::Mat R2 = grayImg(cv::Rect(0, r - h, grayImg.cols, h));
				float S = MI(R1, R2);
				if (S > SV_max(r, 0)) {
					SV_max(r, 0) = S;
					h_max(r, 0) = h;
				}
			}
		}
		printf("\n");

		// output SV_max(x) and h_max(x)
		cvutils::write("SV_max.txt", SV_max);
		cvutils::write("h_max.txt", h_max);
	}

	// initialize IF
	IF = cv::Mat(grayImg.rows, grayImg.cols, CV_32FC4);
	for (int r = 0; r < IF.rows; ++r) {
		for (int c = 0; c < IF.cols; ++c) {
			IF.at<cv::Vec4f>(r, c) = cv::Vec4f(img.at<cv::Vec3b>(r, c)[0], img.at<cv::Vec3b>(r, c)[1], img.at<cv::Vec3b>(r, c)[2], 1.0f);
		}
	}
	outputIF(IF, "IF1.png");

	// find the maximum of SV_max(y)
	float S = 0.0f;
	int y;
	for (int r = 0; r < SV_max.rows; ++r) {
		if (SV_max(r, 0) > S) {
			y = r;
			S = SV_max(r, 0);
		}
	}

	y_set.push_back(y);

	cout << "y: " << y << ", S: " << S << ", h: " << h_max(y, 0) << endl;

	// check splits
	findVerticalSplits(SV_max, h_max, y - h_max(y, 0), h_max(y, 0), S, -1, y_set);
	findVerticalSplits(SV_max, h_max, y - h_max(y, 0), h_max(y, 0), S, 1, y_set);

	sort(y_set.begin(), y_set.end());
	vshrinkIF(IF, y_set, h_max);

	// add the both ends of splits
	y_set.insert(y_set.begin(), y_set[0] - h_max(y_set[0], 0));
	y_set.push_back(y_set.back() + h_max(y_set.back(), 0));

	// visualize S_max_V(y) and h_max(y)
	outputFacadeStructureV(img, SV_max, h_max, y_set, "result.png");
}

void horizontalSplit(const cv::Mat& img, vector<int>& x_set) {
	cv::Mat grayImg;

	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

	cv::Mat_<float> SH_max(1, grayImg.cols, 0.0f);
	cv::Mat_<float> w_max(1, grayImg.cols, 0.0f);

	ifstream in_SH("SH_max.txt");
	ifstream in_w("w_max.txt");
	if (in_SH.good() && in_w.good()) {
		in_SH.close();
		in_w.close();
		SH_max = cvutils::read("SH_max.txt");
		w_max = cvutils::read("w_max.txt");
	}
	else {
		printf("computing");
		for (int c = 0; c < grayImg.cols; ++c) {
			printf("\rcomputing c = %d/%d  ", c, grayImg.cols);

			for (int w = 80; w < 200; ++w) {
				if (c - w < 0 || c + w >= grayImg.cols) continue;

				cv::Mat R1 = grayImg(cv::Rect(c, 0, w, grayImg.rows));
				cv::Mat R2 = grayImg(cv::Rect(c - w, 0, w, grayImg.rows));

				float S = MI(R1, R2);
				if (S > SH_max(0, c)) {
					SH_max(0, c) = S;
					w_max(0, c) = w;
				}
			}
		}
		printf("\n");

		// output SH_max(x) and w_max(x)
		cvutils::write("SH_max.txt", SH_max);
		cvutils::write("w_max.txt", w_max);
	}

	// initialize IF
	cv::Mat IF(grayImg.rows, grayImg.cols, CV_32FC4);
	for (int r = 0; r < IF.rows; ++r) {
		for (int c = 0; c < IF.cols; ++c) {
			IF.at<cv::Vec4f>(r, c) = cv::Vec4f(img.at<cv::Vec3b>(r, c)[0], img.at<cv::Vec3b>(r, c)[1], img.at<cv::Vec3b>(r, c)[2], 1.0f);
		}
	}
	outputIF(IF, "IF11.png");

	// find the maximum of SH_max(x)
	float S = 0.0f;
	int x;
	for (int c = 0; c < SH_max.cols; ++c) {
		if (SH_max(0, c) > S) {
			x = c;
			S = SH_max(0, c);
		}
	}

	x_set.push_back(x);

	cout << "x: " << x << ", S: " << S << ", w: " << w_max(0, x) << endl;

	// check splits
	findHorizontalSplits(SH_max, w_max, x - w_max(0, x), w_max(0, x), S, -1, x_set);
	findHorizontalSplits(SH_max, w_max, x + w_max(0, x), w_max(0, x), S, 1, x_set);

	sort(x_set.begin(), x_set.end());
	hshrinkIF(IF, x_set, w_max);

	// add the both ends of splits
	x_set.insert(x_set.begin(), x_set[0] - w_max(0, x_set[0]));
	x_set.push_back(x_set.back() + w_max(0, x_set.back()));

	// visualize S_max_V(y) and h_max(y)
	outputFacadeStructureH(img, SH_max, w_max, x_set, "result2.png");
}

void computeVerAndHor(const cv::Mat& img, cv::Mat_<float>& Ver, cv::Mat_<float>& Hor) {
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
	Ver = cv::Mat_<float>(grayImg.rows, 1, 0.0f);
	Hor = cv::Mat_<float>(1, grayImg.cols, 0.0f);
	float sigma = 50.0f;
	float beta = 0.1f;
	for (int r = 0; r < grayImg.rows; ++r) {
		for (int rr = 0; rr < grayImg.rows; ++rr) {
			Ver(r, 0) += (ver_xtotal.at<float>(rr, 0) - beta * hor_xtotal.at<float>(rr, 0)) * utils::gause(rr - r, sigma);
		}
	}
	for (int c = 0; c < grayImg.cols; ++c) {
		for (int cc = 0; cc < grayImg.cols; ++cc) {
			Hor(0, c) += (ver_ytotal.at<float>(0, cc) - beta * hor_ytotal.at<float>(0, cc)) * utils::gause(cc - c, sigma);
		}
	}
}

int main() {
	cv::Mat img = cv::imread("../facade/facade.png");

	// vertical split
	vector<int> y_set;
	cv::Mat IF;
	verticalSplit(img, y_set, IF);

	// horizontal split
	vector<int> x_set;
	cv::Mat imgIF;
	createIFImage(IF, imgIF);
	horizontalSplit(imgIF, x_set);

	outputFacadeStructure(img, y_set, x_set, "result3.png");

	return 0;
}