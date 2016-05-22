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
#include <iterator>
#include <opencv2/opencv.hpp>
#include "CVUtils.h"
#include "Utils.h"

using namespace std;

class Subdivision {
public:
	static enum { TYPE_LEFT = 0, TYPE_RIGHT, TYPE_TOP, TYPE_BOTTOM };

public:
	int dir;
	bool dual;
	int dist;

public:
	Subdivision() : dir(0), dual(false), dist(0) {}
	Subdivision(int dir, int dual, int dist) : dir(dir), dual(dual), dist(dist) {}
};

void outputFacadeStructureV(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& h_max, const vector<vector<int>>& y_set, const string& filename) {
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
		for (int j = 0; j < y_set[i].size() - 1; ++j) {
			cout << "Y: " << y_set[i][j] << ", h: " << h_max(y_set[i][j], 0) << endl;

			cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
			cv::rectangle(result, cv::Rect(0, y_set[i][j], img.cols, y_set[i][j + 1] - y_set[i][j]), color, 3);
		}
	}
	cv::imwrite(filename, result);
}

void outputFacadeStructureH(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& w_max, const vector<vector<int>>& x_set, const string& filename) {
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
		for (int j = 0; j < x_set[i].size() - 1; ++j) {
			cout << "X: " << x_set[i][j] << ", w: " << w_max(0, x_set[i][j]) << endl;

			cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
			cv::rectangle(result, cv::Rect(x_set[i][j], 0, x_set[i][j + 1] - x_set[i][j], img.rows), color, 3);
		}
	}
	cv::imwrite(filename, result);
}

void outputFacadeStructure(const cv::Mat& img, const vector<vector<int>>& y_set, const vector<vector<int>>& x_set, const string& filename) {
	cv::Mat result = img.clone();


	for (int i = 0; i < y_set.size(); ++i) {
		for (int j = 0; j < y_set[i].size(); ++j) {
			cv::line(result, cv::Point(0, y_set[i][j]), cv::Point(img.cols, y_set[i][j]), cv::Scalar(0, 0, 255), 3);
		}
	}
	for (int i = 0; i < x_set.size(); ++i) {
		for (int j = 0; j < x_set[i].size(); ++j) {
			cv::line(result, cv::Point(x_set[i][j], 0), cv::Point(x_set[i][j], img.rows), cv::Scalar(0, 0, 255), 3);
		}
	}
	cv::imwrite(filename, result);
}

void outputTileStructure(const cv::Mat& tile, const cv::Mat_<float>& Ver, const cv::Mat_<float>& Hor, const string& filename) {
	cv::Mat result(tile.rows + 100, tile.cols + 100, CV_8UC3, cv::Scalar(255, 255, 255));

	// get the maximum value of Ver(y) and Hor(x)
	float max_Ver = cvutils::max(Ver);
	float min_Ver = cvutils::min(Ver);
	float max_Hor = cvutils::max(Hor);
	float min_Hor = cvutils::min(Hor);

	for (int r = 0; r < result.rows; ++r) {
		for (int c = 0; c < result.cols; ++c) {
			if (r < tile.rows && c < tile.cols) {
				result.at<cv::Vec3b>(r, c) = tile.at<cv::Vec3b>(r, c);
			}
			else if (r < tile.rows - 1) {
				int v1 = (min(Ver(r, 0), Ver(r + 1, 0)) - min_Ver) / (max_Ver - min_Ver) * 100;
				int v2 = (max(Ver(r, 0), Ver(r + 1, 0)) - min_Ver) / (max_Ver - min_Ver) * 100;
				if (c - tile.cols >= v1 && c - tile.cols <= v2) {
					result.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
				}
			}
			else if (c < tile.cols - 1) {
				int v1 = (min(Hor(0, c), Hor(0, c + 1)) - min_Hor) / (max_Hor - min_Hor) * 100;
				int v2 = (max(Hor(0, c), Hor(0, c + 1)) - min_Hor) / (max_Hor - min_Hor) * 100;
				if (r - tile.rows >= v1 && r - tile.rows <= v2) {
					result.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
				}
			}
		}
	}

	cv::imwrite(filename, result);
}

void drawSubdivisionOnTile(cv::Mat& tile, const vector<Subdivision>& subdivisions, const cv::Scalar& color, int thickness) {
	int x1 = 0;
	int x2 = tile.cols - 1;
	int y1 = 0;
	int y2 = tile.rows - 1;

	for (int i = 0; i < subdivisions.size(); ++i) {
		if (subdivisions[i].dir == Subdivision::TYPE_LEFT) {
			x1 += subdivisions[i].dist;
			cv::line(tile, cv::Point(x1, y1), cv::Point(x1, y2), color, thickness);
			if (subdivisions[i].dual) {
				x2 -= subdivisions[i].dist;
				cv::line(tile, cv::Point(x2, y1), cv::Point(x2, y2), color, thickness);
			}
		}
		else if (subdivisions[i].dir == Subdivision::TYPE_RIGHT) {
			x2 -= subdivisions[i].dist;
			cv::line(tile, cv::Point(x2, y1), cv::Point(x2, y2), color, thickness);
			if (subdivisions[i].dual) {
				x1 -= subdivisions[i].dist;
				cv::line(tile, cv::Point(x1, y1), cv::Point(x1, y2), color, thickness);
			}
		}
		else if (subdivisions[i].dir == Subdivision::TYPE_TOP) {
			y1 += subdivisions[i].dist;
			cv::line(tile, cv::Point(x1, y1), cv::Point(x2, y1), color, thickness);
			if (subdivisions[i].dual) {
				y2 -= subdivisions[i].dist;
				cv::line(tile, cv::Point(x1, y2), cv::Point(x2, y2), color, thickness);
			}
		}
		else if (subdivisions[i].dir == Subdivision::TYPE_BOTTOM) {
			y2 -= subdivisions[i].dist;
			cv::line(tile, cv::Point(x1, y2), cv::Point(x2, y2), color, thickness);
			if (subdivisions[i].dual) {
				y1 -= subdivisions[i].dist;
				cv::line(tile, cv::Point(x1, y1), cv::Point(x2, y1), color, thickness);
			}
		}
	}
}

void outputFacadeAndTileStructure(const cv::Mat& img, const vector<vector<int>>& y_set, const vector<vector<int>>& x_set, const vector<vector<vector<Subdivision>>>& subdivisions, const string& filename) {
	cv::Mat img2 = img.clone();

	// visualize the subdivision of tiles
	for (int i = 0; i < y_set.size(); ++i) {
		for (int j = 0; j < y_set[i].size() - 1; ++j) {
			int y1 = y_set[i][j];
			int y2 = y_set[i][j + 1];

			for (int k = 0; k < x_set.size(); ++k) {
				for (int l = 0; l < x_set[k].size() - 1; ++l) {
					int x1 = x_set[k][l];
					int x2 = x_set[k][l + 1];

					int u1 = x1;
					int u2 = x2;
					int v1 = y1;
					int v2 = y2;

					cv::Mat tile(img2, cv::Rect(x1, y1, x2 - x1 - 1, y2 - y1 - 1));
					drawSubdivisionOnTile(tile, subdivisions[i][k], cv::Scalar(255, 0, 0), 3);
				}
			}
		}
	}

	// visualize the subdivision of facade
	for (int i = 0; i < y_set.size(); ++i) {
		for (int j = 0; j < y_set[i].size(); ++j) {
			cv::line(img2, cv::Point(0, y_set[i][j]), cv::Point(img2.cols - 1, y_set[i][j]), cv::Scalar(0, 0, 255), 3);
		}
	}
	for (int i = 0; i < x_set.size(); ++i) {
		for (int j = 0; j < x_set[i].size(); ++j) {
			cv::line(img2, cv::Point(x_set[i][j], 0), cv::Point(x_set[i][j], img2.rows - 1), cv::Scalar(0, 0, 255), 3);
		}
	}

	cv::imwrite(filename, img2);
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
void vshrinkIF(cv::Mat& IF, const vector<vector<int>>& y_set, const cv::Mat_<float>& h_max) {
	for (int i = y_set.size() - 1; i >= 0; --i) {
		for (int j = y_set[i].size() - 2; j >= 1; --j) {
			vshrinkIF(IF, y_set[i][j], h_max(y_set[i][j], 0));
		}
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
void hshrinkIF(cv::Mat& IF, const vector<vector<int>>& x_set, const cv::Mat_<float>& w_max) {
	for (int i = x_set.size() - 1; i >= 0; --i) {
		for (int j = x_set[i].size() - 2; j >= 1; --j) {
			hshrinkIF(IF, x_set[i][j], w_max(0, x_set[i][j]));
		}
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

void findVerticalSplits(const cv::Mat_<float>& S_max, const cv::Mat_<float>& h_max, int y, int h, float tau_max, int dir, vector<vector<int>>& y_set) {
	while (true) {
		int next_y;
		float S = findAdjacentVerticalSplit(S_max, h_max, y, h, next_y);

		cout << "y: " << next_y << ", S: " << S << ", h: " << h_max(next_y, 0) << endl;

		if (S >= tau_max * 0.75f) {
			if (dir == -1) {
				y_set.front().insert(y_set.front().begin(), next_y);
			}
			else {
				y_set.back().push_back(next_y);
			}
			y = next_y + h * dir;
		}
		else {
			if (dir == -1) {
				y_set.front().insert(y_set.front().begin(), y);
			}
			else {
				y_set.back().push_back(y);
			}

			cout << " --> not good" << endl;

			S = findNextVerticalSplit(S_max, h_max, y, dir, next_y);
			if (S >= tau_max * 0.75f) {
				if (dir == -1) {
					y_set.insert(y_set.begin(), vector<int>());
					y_set.front().push_back(next_y);
					y_set.front().push_back(y);
				}
				else {
					y_set.push_back(vector<int>());
					y_set.back().push_back(y);
					y_set.back().push_back(next_y);
				}
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

void findHorizontalSplits(const cv::Mat_<float>& S_max, const cv::Mat_<float>& w_max, int x, int w, float tau_max, int dir, vector<vector<int>>& x_set) {
	while (true) {
		int next_x;
		float S = findAdjacentHorizontalSplit(S_max, w_max, x, w, next_x);

		cout << "x: " << next_x << ", S: " << S << ", w: " << w_max(0, next_x) << endl;

		if (S >= tau_max * 0.75f) {
			if (dir == -1) {
				x_set.front().insert(x_set.front().begin(), next_x);
			}
			else {
				x_set.back().push_back(next_x);
			}
			x = next_x + w * dir;
		}
		else {
			if (dir == -1) {
				x_set.front().insert(x_set.front().begin(), x);
			}
			else {
				x_set.back().push_back(x);
			}

			cout << " --> not good" << endl;

			S = findNextHorizontalSplit(S_max, w_max, x, dir, next_x);
			if (S >= tau_max * 0.75f) {
				if (dir == -1) {
					x_set.insert(x_set.begin(), vector<int>());
					x_set.front().push_back(next_x);
					x_set.front().push_back(x);
				}
				else {
					x_set.push_back(vector<int>());
					x_set.back().push_back(x);
					x_set.back().push_back(next_x);
				}
				x = next_x + w_max(next_x, 0) * dir;
			}
			else {
				break;
			}
		}
	}

	cout << "Terminated." << endl;
}

void verticalSplit(const cv::Mat& img, vector<vector<int>>& y_set, cv::Mat& IF) {
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

	y_set.push_back(vector<int>(1, y));

	cout << "y: " << y << ", S: " << S << ", h: " << h_max(y, 0) << endl;

	// check splits
	findVerticalSplits(SV_max, h_max, y - h_max(y, 0), h_max(y, 0), S, -1, y_set);
	findVerticalSplits(SV_max, h_max, y + h_max(y, 0), h_max(y, 0), S, 1, y_set);

	cout << "DEbug!!!!!!!!!!!!!!" << endl;
	for (int i = 0; i < y_set.size(); ++i) {
		for (int j = 0; j < y_set[i].size(); ++j) {
			cout << y_set[i][j] << ", ";
		}
	}
	cout << endl;

	//sort(y_set.begin(), y_set.end());
	vshrinkIF(IF, y_set, h_max);

	// visualize S_max_V(y) and h_max(y)
	outputFacadeStructureV(img, SV_max, h_max, y_set, "result.png");
}

void horizontalSplit(const cv::Mat& img, vector<vector<int>>& x_set) {
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

	x_set.push_back(vector<int>(1, x));

	cout << "x: " << x << ", S: " << S << ", w: " << w_max(0, x) << endl;

	// check splits
	findHorizontalSplits(SH_max, w_max, x - w_max(0, x), w_max(0, x), S, -1, x_set);
	findHorizontalSplits(SH_max, w_max, x + w_max(0, x), w_max(0, x), S, 1, x_set);

	//sort(x_set.begin(), x_set.end());
	hshrinkIF(IF, x_set, w_max);

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
	cv::reduce(hor, hor_ytotal, 0, cv::REDUCE_SUM);

	// compute Ver(y) and Hor(x) according to Equation (4)
	Ver = cv::Mat_<float>(grayImg.rows, 1, 0.0f);
	Hor = cv::Mat_<float>(1, grayImg.cols, 0.0f);
	float sigma = 5.0f;// 50.0f;
	float beta = 0.1f;
	for (int r = 0; r < grayImg.rows; ++r) {
		for (int rr = 0; rr < grayImg.rows; ++rr) {
			Ver(r, 0) += (ver_xtotal.at<float>(rr, 0) - beta * hor_xtotal.at<float>(rr, 0)) * utils::gause(rr - r, sigma);
		}
	}
	for (int c = 0; c < grayImg.cols; ++c) {
		for (int cc = 0; cc < grayImg.cols; ++cc) {
			Hor(0, c) += (hor_ytotal.at<float>(0, cc) - beta * ver_ytotal.at<float>(0, cc)) * utils::gause(cc - c, sigma);
		}
	}
}

/**
 * tileを分割し、分割方向、分割タイプ、ボーダーからの距離を返却する。
 * 分割しない場合はfalseを返却する。
 */
bool subdivideTile(cv::Mat& tile, Subdivision& subdivide, int min_size) {
	if (tile.cols < min_size || tile.rows < min_size) return false;

	cv::imwrite("tile.png", tile);

	cv::Mat_<float> Ver;
	cv::Mat_<float> Hor;
	computeVerAndHor(tile, Ver, Hor);

	// visualize Ver(y) and Hor(x)
	outputTileStructure(tile, Ver, Hor, "tile2.png");


	// find the local minima of Ver(y) and Hor(x)
	vector<int> y_set;
	for (int r = min_size; r < Ver.rows - min_size; ++r) {
		if (Ver(r, 0) < Ver(r - 1, 0) && Ver(r, 0) < Ver(r + 1, 0)) {
			y_set.push_back(r);
		}
	}
	vector<int> x_set;
	for (int c = min_size; c < Hor.cols - min_size; ++c) {
		if (Hor(0, c) < Hor(0, c - 1) && Hor(0, c) < Hor(0, c + 1)) {
			x_set.push_back(c);
		}
	}

	if (x_set.size() == 0 && y_set.size() == 0) return false;

	// find the split closest to the boundary
	subdivide.dist = numeric_limits<int>::max();
	int index = -1;
	if (x_set.size() > 0) {
		if (x_set[0] < tile.cols - x_set.back() - 1) {
			subdivide.dir = Subdivision::TYPE_LEFT;
			subdivide.dist = x_set[0];
			index = 0;
		}
		else {
			subdivide.dir = Subdivision::TYPE_RIGHT;
			subdivide.dist = tile.cols - x_set.back() - 1;
			index = x_set.size() - 1;
		}
	}
	if (y_set.size() > 0) {
		if (y_set[0] < subdivide.dist && y_set[0] < tile.rows - y_set.back() - 1) {
			subdivide.dir = Subdivision::TYPE_TOP;
			subdivide.dist = y_set[0];
			index = 0;
		}
		else if (tile.rows - y_set.back() - 1 < subdivide.dist) {
			subdivide.dir = Subdivision::TYPE_BOTTOM;
			subdivide.dist = tile.rows - y_set.back() - 1;
			index = y_set.size() - 1;
		}
	}

	// find the dual correspondence
	if (subdivide.dir == Subdivision::TYPE_LEFT) {
		for (int i = 0; i < x_set.size(); ++i) {
			if (i == index) continue;

			if (abs(tile.cols - x_set[i] - 1 - subdivide.dist) < 3) {
				subdivide.dual = true;
				break;
			}
		}
	}
	else if (subdivide.dir == Subdivision::TYPE_RIGHT) {
		for (int i = 0; i < x_set.size(); ++i) {
			if (i == index) continue;

			if (abs(x_set[i] - subdivide.dist) < 3) {
				subdivide.dual = true;
				break;
			}
		}
	}
	else if (subdivide.dir == Subdivision::TYPE_TOP) {
		for (int i = 0; i < y_set.size(); ++i) {
			if (i == index) continue;

			if (abs(tile.rows - y_set[i] - 1 - subdivide.dist) < 3) {
				subdivide.dual = true;
				break;
			}
		}
	}
	else if (subdivide.dir == Subdivision::TYPE_BOTTOM) {
		for (int i = 0; i < y_set.size(); ++i) {
			if (i == index) continue;

			if (abs(y_set[i] - subdivide.dist) < 3) {
				subdivide.dual = true;
				break;
			}
		}
	}

	return true;
}

/**
 * 領域(x1,y1)-(x2,y2)を分割し、領域を更新する。
 */
void updateRegion(const Subdivision& subdivision, int& x1, int& y1, int& x2, int& y2) {
	if (subdivision.dir == Subdivision::TYPE_LEFT) {
		x1 += subdivision.dist;
		if (subdivision.dual) {
			x2 -= subdivision.dist;
		}
	}
	else if (subdivision.dir == Subdivision::TYPE_RIGHT) {
		x2 -= subdivision.dist;
		if (subdivision.dual) {
			x1 -= subdivision.dist;
		}
	}
	else if (subdivision.dir == Subdivision::TYPE_TOP) {
		y1 += subdivision.dist;
		if (subdivision.dual) {
			y2 -= subdivision.dist;
		}
	}
	else if (subdivision.dir == Subdivision::TYPE_BOTTOM) {
		y2 -= subdivision.dist;
		if (subdivision.dual) {
			y1 -= subdivision.dist;
		}
	}
}

/**
 * facadeの各tileを1ステップ、細かく分割し、subdivisionsに分割方法を追加する。
 * 1つのタイルも分割しない場合はfalseを返却する。
 */
bool subdivideFacadeTilesByOneStep(cv::Mat& img, vector<vector<int>>& y_set, vector<vector<int>>& x_set, vector<vector<vector<Subdivision>>>& subdivisions) {
	vector<vector<vector<Subdivision>>> votes(y_set.size());

	for (int i = 0; i < y_set.size(); ++i) {
		for (int j = 0; j < y_set[i].size() - 1; ++j) {
			int y1 = y_set[i][j];
			int y2 = y_set[i][j + 1];

			votes[i].resize(x_set.size());
			for (int k = 0; k < x_set.size(); ++k) {
				for (int l = 0; l < x_set[k].size() - 1; ++l) {
					int x1 = x_set[k][l];
					int x2 = x_set[k][l + 1];

					int u1 = x1;
					int u2 = x2;
					int v1 = y1;
					int v2 = y2;

					for (int s = 0; s < subdivisions[i][k].size(); ++s) {
						updateRegion(subdivisions[i][k][s], u1, v1, u2, v2);
					}

					cv::Mat tile(img, cv::Rect(u1, v1, u2 - u1 - 1, v2 - v1 - 1));
					Subdivision subdivide;
					if (subdivideTile(tile, subdivide, max(x2 - x1, y2 - y1) * 0.2)) {
						votes[i][k].push_back(subdivide);
					}
				}
			}
		}
	}

	bool subdivided = false;

	// choose the maximum vote for each type of tile
	for (int i = 0; i < votes.size(); ++i) {
		for (int j = 0; j < votes[i].size(); ++j) {
			if (votes[i][j].size() == 0) continue;

			subdivided = true;
						
			float sigma = 3.0f;

			// find the maximum vote for horizontal split
			float x_vote_max;
			Subdivision x_subdivide(0, false, 0);
			{
				vector<float> histogram(x_set[j][1] - x_set[j][0], 0.0f);

				for (int s = 0; s < votes[i][j].size(); ++s) {
					if (votes[i][j][s].dir == Subdivision::TYPE_TOP || votes[i][j][s].dir == Subdivision::TYPE_BOTTOM) continue;

					int dist;
					if (votes[i][j][s].dir == Subdivision::TYPE_LEFT) {
						dist = votes[i][j][s].dist;
					}
					else if (votes[i][j][s].dir == Subdivision::TYPE_RIGHT) {
						dist = x_set[j][1] - x_set[j][0] - votes[i][j][s].dist;
					}

					for (int c = 0; c < x_set[j][1] - x_set[j][0]; ++c) {
						histogram[c] += utils::gause(c - dist, sigma);
					}
				}

				x_vote_max = *max_element(histogram.begin(), histogram.end());
				x_subdivide.dist = distance(histogram.begin(), max_element(histogram.begin(), histogram.end()));

				// select the type
				int min_dist = numeric_limits<int>::max();
				for (int s = 0; s < votes[i][j].size(); ++s) {
					if (votes[i][j][s].dir == Subdivision::TYPE_LEFT) {
						int d = abs(votes[i][j][s].dist - x_subdivide.dist);
						if (d < min_dist) {
							min_dist = d;
							x_subdivide = votes[i][j][s];
						}
					}
					else if (votes[i][j][s].dir == Subdivision::TYPE_RIGHT) {
						int d = abs(x_set[j][1] - x_set[j][0] - votes[i][j][s].dist - x_subdivide.dist);
						if (d < min_dist) {
							min_dist = d;
							x_subdivide = votes[i][j][s];
						}
					}
				}
			}

			// find the maximum vote for vertical split
			float y_vote_max;
			Subdivision y_subdivide(1, false, 0);
			{
				vector<float> histogram(y_set[j][1] - y_set[j][0], 0.0f);

				for (int s = 0; s < votes[i][j].size(); ++s) {
					if (votes[i][j][s].dir == Subdivision::TYPE_LEFT || votes[i][j][s].dir == Subdivision::TYPE_RIGHT) continue;

					int dist;
					if (votes[i][j][s].dir == Subdivision::TYPE_TOP) {
						dist = votes[i][j][s].dist;
					}
					else if (votes[i][j][s].dir == Subdivision::TYPE_BOTTOM) {
						dist = y_set[j][1] - y_set[j][0] - votes[i][j][s].dist;
					}

					for (int r = 0; r < y_set[j][1] - y_set[j][0]; ++r) {
						histogram[r] += utils::gause(r - dist, sigma);
					}
				}

				y_vote_max = *max_element(histogram.begin(), histogram.end());
				y_subdivide.dist = distance(histogram.begin(), max_element(histogram.begin(), histogram.end()));

				// select the type
				int min_dist = numeric_limits<int>::max();
				for (int s = 0; s < votes[i][j].size(); ++s) {
					if (votes[i][j][s].dir == Subdivision::TYPE_TOP) {
						int d = abs(votes[i][j][s].dist - y_subdivide.dist);
						if (d < min_dist) {
							min_dist = d;
							y_subdivide = votes[i][j][s];
						}
					}
					else if (votes[i][j][s].dir == Subdivision::TYPE_BOTTOM) {
						int d = abs(y_set[j][1] - y_set[j][0] - votes[i][j][s].dist - y_subdivide.dist);
						if (d < min_dist) {
							min_dist = d;
							y_subdivide = votes[i][j][s];
						}
					}
				}
			}

			if (x_vote_max > y_vote_max) {
				subdivisions[i][j].push_back(x_subdivide);
			}
			else {
				subdivisions[i][j].push_back(y_subdivide);
			}
		}
	}

	return subdivided;
}

void subdivideFacadeTiles(cv::Mat& img, vector<vector<int>>& y_set, vector<vector<int>>& x_set, vector<vector<vector<Subdivision>>>& subdivisions) {
	subdivisions.resize(y_set.size());
	for (int i = 0; i < y_set.size(); ++i) {
		subdivisions[i].resize(x_set.size());
	}

	// 各tileを細かく分割する
	while (true) {
		if (!subdivideFacadeTilesByOneStep(img, y_set, x_set, subdivisions)) break;

		outputFacadeAndTileStructure(img, y_set, x_set, subdivisions, "result4b.png");
	}
}

int main() {
	cv::Mat img = cv::imread("../facade/facade.png");

	// vertical split
	vector<vector<int>> y_set;
	cv::Mat IF;
	verticalSplit(img, y_set, IF);

	// add the image boundary as split
	if (y_set[0][0] > 0) {
		y_set.insert(y_set.begin(), vector<int>());
		y_set.front().push_back(0);
		y_set.front().push_back(y_set[1][0]);
	}
	if (y_set.back().back() < img.rows - 1) {
		y_set.push_back(vector<int>());
		y_set.back().push_back(y_set[y_set.size() - 2].back());
		y_set.back().push_back(img.rows - 1);
	}

	// horizontal split
	vector<vector<int>> x_set;
	cv::Mat imgIF;
	createIFImage(IF, imgIF);
	horizontalSplit(imgIF, x_set);

	// add the image boundary as split
	if (x_set[0][0] > 0) {
		x_set.insert(x_set.begin(), vector<int>());
		x_set.front().push_back(0);
		x_set.front().push_back(x_set[1][0]);
	}
	if (x_set.back().back() < img.cols - 1) {
		x_set.push_back(vector<int>());
		x_set.back().push_back(x_set[x_set.size() - 2].back());
		x_set.back().push_back(img.cols - 1);
	}

	outputFacadeStructure(img, y_set, x_set, "result3.png");
	
	vector<vector<vector<Subdivision>>> subdivisions;
	subdivideFacadeTiles(img, y_set, x_set, subdivisions);
	outputFacadeAndTileStructure(img, y_set, x_set, subdivisions, "result4.png");

	return 0;
}