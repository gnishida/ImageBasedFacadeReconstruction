/**
* 論文Image-based procedural modeling of facadesに基づいて、
* Facadeを分割する。
* 5/24に第一回の報告をしたが、いろいろ難しいことが分かった。
* まず、MIではうまくsymmetryをキャプチャできないケースがある。
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
#include <time.h>

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

/**
 * Facade画像と合わせて、S_max(y)とh_max(y)を画像として保存する。
 * 論文 Fig 5に相当する画像。
 *
 * @param img		Facade画像
 * @param S_max		S_max
 * @param h_max		h_max
 * @param filename	output file name
 */
void outputFacadeStructureV(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& h_max, const vector<int>& y_set, const string& filename, int lineWidth) {
	float max_S = cvutils::max(S_max);
	float min_S = cvutils::min(S_max);
	float max_h = cvutils::max(h_max);
	float min_h = cvutils::min(h_max);

	int graphSize = img.rows * 0.25;
	int margin = graphSize * 0.2;
	cv::Mat result(img.rows, img.cols + graphSize + max_h + margin * 3, CV_8UC3, cv::Scalar(255, 255, 255));

	// copy img to result
	cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
	img.copyTo(roi);

	// draw S_max
	for (int r = 0; r < img.rows - 1; ++r) {
		int x1 = img.cols + margin + (S_max(r, 0) - min_S) / (max_S - min_S) * graphSize;
		int x2 = img.cols + margin + (S_max(r + 1, 0) - min_S) / (max_S - min_S) * graphSize;

		cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
	}

	// draw h_max
	for (int r = 0; r < img.rows - 1; ++r) {
		int x1 = img.cols + graphSize + margin * 2 + h_max(r, 0);
		int x2 = img.cols + graphSize + margin * 2 + h_max(r + 1, 0);

		cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
	}

	// draw the horizontal split lines
	for (int i = 0; i < y_set.size(); ++i) {
		cv::line(result, cv::Point(0, y_set[i]), cv::Point(img.cols - 1, y_set[i]), cv::Scalar(0, 0, 255), lineWidth);
	}

	cv::imwrite(filename, result);
}

void outputFacadeStructureV(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& h_max, const string& filename, int lineWidth) {
	vector<int> y_set;
	outputFacadeStructureV(img, S_max, h_max, y_set, filename, lineWidth);
}

void outputFacadeStructureH(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& w_max, const vector<int>& x_set, const string& filename, int lineWidth) {
	float max_S = cvutils::max(S_max);
	float min_S = cvutils::min(S_max);
	float max_w = cvutils::max(w_max);
	float min_w = cvutils::min(w_max);

	int graphSize = max(80.0, img.rows * 0.25);
	int margin = graphSize * 0.2;
	cv::Mat result(img.rows + graphSize + max_w + margin * 3, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));

	// copy img to result
	cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
	img.copyTo(roi);

	// draw S_max
	for (int c = 0; c < img.cols - 1; ++c) {
		int y1 = img.rows + margin + (S_max(0, c) - min_S) / (max_S - min_S) * graphSize;
		int y2 = img.rows + margin + (S_max(0, c + 1) - min_S) / (max_S - min_S) * graphSize;

		cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
	}

	// draw w_max
	for (int c = 0; c < img.cols - 1; ++c) {
		int y1 = img.rows + graphSize + margin * 2 + w_max(0, c);
		int y2 = img.rows + graphSize + margin * 2 + w_max(0, c + 1);

		cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
	}

	// draw the vertical split lines
	for (int i = 0; i < x_set.size(); ++i) {
		cv::line(result, cv::Point(x_set[i], 0), cv::Point(x_set[i], img.rows - 1), cv::Scalar(0, 0, 255), lineWidth);
	}

	cv::imwrite(filename, result);
}

void outputFacadeStructureH(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& w_max, const string& filename, int lineWidth) {
	vector<int> x_set;
	outputFacadeStructureH(img, S_max, w_max, x_set, filename, lineWidth);
}

void outputFacadeStructure(const cv::Mat& img, const cv::Mat_<float>& SV_max, const cv::Mat_<float>& h_max, const cv::Mat_<float>& SH_max, const cv::Mat_<float>& w_max, const vector<int>& y_set, const vector<int>& x_set, const string& filename, int lineWidth) {
	float max_SV = cvutils::max(SV_max);
	float min_SV = cvutils::min(SV_max);
	float max_h = cvutils::max(h_max);
	float min_h = cvutils::min(h_max);
	float max_SH = cvutils::max(SH_max);
	float min_SH = cvutils::min(SH_max);
	float max_w = cvutils::max(w_max);
	float min_w = cvutils::min(w_max);

	int graphSize = max(80.0, img.rows * 0.25);
	int margin = graphSize * 0.2;
	cv::Mat result(img.rows + graphSize + max_w + margin * 3, img.cols + graphSize + max_h + margin * 3, CV_8UC3, cv::Scalar(255, 255, 255));

	// copy img to result
	cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
	img.copyTo(roi);

	// draw SV_max
	for (int r = 0; r < img.rows - 1; ++r) {
		int x1 = img.cols + margin + (SV_max(r, 0) - min_SV) / (max_SV - min_SV) * graphSize;
		int x2 = img.cols + margin + (SV_max(r + 1, 0) - min_SV) / (max_SV - min_SV) * graphSize;

		cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
	}

	// draw h_max
	for (int r = 0; r < img.rows - 1; ++r) {
		int x1 = img.cols + graphSize + margin * 2 + h_max(r, 0);
		int x2 = img.cols + graphSize + margin * 2 + h_max(r + 1, 0);

		cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
	}

	// draw SH_max
	for (int c = 0; c < img.cols - 1; ++c) {
		int y1 = img.rows + margin + (SH_max(0, c) - min_SH) / (max_SH - min_SH) * graphSize;
		int y2 = img.rows + margin + (SH_max(0, c + 1) - min_SH) / (max_SH - min_SH) * graphSize;

		cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
	}

	// draw w_max
	for (int c = 0; c < img.cols - 1; ++c) {
		int y1 = img.rows + graphSize + margin * 2 + w_max(0, c);
		int y2 = img.rows + graphSize + margin * 2 + w_max(0, c + 1);

		cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
	}

	// draw the horizontal split lines
	for (int i = 0; i < y_set.size(); ++i) {
		cv::line(result, cv::Point(0, y_set[i]), cv::Point(img.cols - 1, y_set[i]), cv::Scalar(0, 0, 255), lineWidth);
	}

	// draw the vertical split lines
	for (int i = 0; i < x_set.size(); ++i) {
		cv::line(result, cv::Point(x_set[i], 0), cv::Point(x_set[i], img.rows - 1), cv::Scalar(0, 0, 255), lineWidth);
	}

	cv::imwrite(filename, result);
}

void outputFacadeStructure(const cv::Mat& img, const cv::Mat_<float>& SV_max, const cv::Mat_<float>& h_max, const cv::Mat_<float>& SH_max, const cv::Mat_<float>& w_max, const string& filename, int lineWidth) {
	vector<int> x_set;
	vector<int> y_set;
	outputFacadeStructure(img, SV_max, h_max, SH_max, w_max, y_set, x_set, filename, lineWidth);
}

void outputFacadeStructure(const cv::Mat& img, const vector<int>& y_set, const vector<int>& x_set, const string& filename, int lineWidth) {
	cv::Mat result = img.clone();

	for (int i = 0; i < y_set.size(); ++i) {
		cv::line(result, cv::Point(0, y_set[i]), cv::Point(img.cols, y_set[i]), cv::Scalar(0, 0, 255), lineWidth);
	}
	for (int i = 0; i < x_set.size(); ++i) {
		cv::line(result, cv::Point(x_set[i], 0), cv::Point(x_set[i], img.rows), cv::Scalar(0, 0, 255), lineWidth);
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

void outputFacadeAndWindows(const cv::Mat& img, const vector<int>& y_split, const vector<int>& x_split, const vector<vector<cv::Rect>>& window_rects, const string& filename) {
	cv::Mat result = img.clone();
	for (int i = 0; i < y_split.size(); ++i) {
		cv::line(result, cv::Point(0, y_split[i]), cv::Point(result.cols - 1, y_split[i]), cv::Scalar(0, 0, 255), 1);
	}
	for (int i = 0; i < x_split.size(); ++i) {
		cv::line(result, cv::Point(x_split[i], 0), cv::Point(x_split[i], result.rows - 1), cv::Scalar(0, 0, 255), 1);
	}
	for (int i = 0; i < y_split.size() - 1; ++i) {
		for (int j = 0; j < x_split.size() - 1; ++j) {
			if (window_rects[i][j].width > 0 && window_rects[i][j].height > 0) {
				cv::rectangle(result, cv::Rect(x_split[j] + window_rects[i][j].x, y_split[i] + window_rects[i][j].y, window_rects[i][j].width, window_rects[i][j].height), cv::Scalar(255, 0, 0), 1);
			}
		}
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

/**
 * 2つの領域の類似度を返却する。
 *
 * @param R1		領域1 (1-channel image)
 * @param R2		領域2 (1-channel image)
 * @return			類似度
 */
float MI(const cv::Mat& R1, const cv::Mat& R2) {
	cv::Mat norm_R1;
	cv::Mat norm_R2;

	//////////////////////////////////////////////////////////////////////////////////
	// 実験：領域をnormalizeしてみる
#if 0
	int max_R1 = cvutils::max(R1);
	int min_R1 = cvutils::min(R1);
	int max_R2 = cvutils::max(R2);
	int min_R2 = cvutils::min(R2);
	R1.convertTo(norm_R1, R1.type(), 255.0f / (max_R1 - min_R1), -min_R1 * 255.0f / (max_R1 - min_R1));
	R2.convertTo(norm_R2, R2.type(), 255.0f / (max_R2 - min_R2), -min_R2 * 255.0f / (max_R2 - min_R2));
	return expf(-cvutils::msd(norm_R1, norm_R2) * 0.001f);
#endif
	//////////////////////////////////////////////////////////////////////////////////

	return expf(-cvutils::msd(R1, R2) * 0.001f);

	/*
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
	*/
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
* Facade画像のS_max(y)、h_max(y)を計算する。
*
* @param img		Facade画像 (1-channel image)
* @param SV_max		S_max(y)
* @param h_max		h_max(y)
* @param h_range	range of h
*/
void computeSV(const cv::Mat& img, cv::Mat_<float>& SV_max, cv::Mat_<float>& h_max, const pair<int, int>& h_range) {
	SV_max = cv::Mat_<float>(img.rows, 1, 0.0f);
	h_max = cv::Mat_<float>(img.rows, 1, 0.0f);

#if 0
	ifstream in_SV_max("SV_max.txt");
	ifstream in_h("h_max.txt");
	if (in_SV_max.good() && in_h.good()) {
		in_SV_max.close();
		in_h.close();
		SV_max = cvutils::read("SV_max.txt");
		h_max = cvutils::read("h_max.txt");
	}
	else {
#endif
		printf("computing");
		for (int r = 0; r < img.rows; ++r) {
			printf("\rcomputing r = %d/%d  ", r, img.rows);

			cv::Mat_<float> SV(img.rows, 1, 0.0f);

			for (int h = h_range.first; h <= h_range.second; ++h) {
				if (r - h < 0 || r + h >= img.rows) continue;

				cv::Mat R1 = img(cv::Rect(0, r, img.cols, h));
				cv::Mat R2 = img(cv::Rect(0, r - h, img.cols, h));
				SV(h, 0) = MI(R1, R2);
				
				if (SV(h, 0) > SV_max(r, 0)) {
					SV_max(r, 0) = SV(h, 0);
					h_max(r, 0) = h;
				}
			}
		}
		printf("\n");

#if 0
		// output SV_max(x) and h_max(x)
		cvutils::write("SV_max.txt", SV_max);
		cvutils::write("h_max.txt", h_max);
	}
#endif

	////////////////////////////////////////////////////////////////////////
	// DEBUG
#if 0
	{
		cv::Mat_<float> histograms(img.rows, 1, 0.0f);
		int y0 = 150;
		for (int r = 0; r < img.rows; ++r) {
			if (abs(r - y0) < 10) continue;

			if (r < y0) {
				int h = y0 - r;
				if (y0 + h >= img.rows) continue;
				cv::Mat R1 = img(cv::Rect(0, r, img.cols, h));
				cv::Mat R2 = img(cv::Rect(0, r + h, img.cols, h));
				histograms(r, 0) += MI(R1, R2);
			}
			else {
				int h = r - y0;
				if (y0 - h < 0) continue;
				cv::Mat R1 = img(cv::Rect(0, y0 - h, img.cols, h));
				cv::Mat R2 = img(cv::Rect(0, y0, img.cols, h));
				histograms(r, 0) += MI(R1, R2);
			}
		}

		cvutils::outputImageWithVerticalGraph(img, histograms, "temp.png");

		float max_value = 0.0f;
		int max_index = -1;
		for (int i = 0; i < histograms.rows; ++i) {
			if (histograms(i, 0) > max_value) {
				max_value = histograms(i, 0);
				max_index = i;
			}
		}

		if (max_index < y0) {
			int h = y0 - max_index;
			cv::Mat R1 = img(cv::Rect(0, max_index, img.cols, h));
			cv::Mat R2 = img(cv::Rect(0, max_index + h, img.cols, h));
			cv::imwrite("R1.png", R1);
			cv::imwrite("R2.png", R2);
			cout << "MSD: " << cvutils::msd(R1, R2) << endl;
		}
		else {
			int h = max_index - y0;
			cv::Mat R1 = img(cv::Rect(0, y0 - h, img.cols, h));
			cv::Mat R2 = img(cv::Rect(0, y0, img.cols, h));
			cv::imwrite("R1.png", R1);
			cv::imwrite("R2.png", R2);
			cout << "MSD: " << cvutils::msd(R1, R2) << endl;
		}
	}
#endif
	////////////////////////////////////////////////////////////////////////
}

/**
 * Facade画像のS_max(x)、w_max(x)を計算する。
 *
 * @param img		Facade画像 (1-channel image)
 * @param SH_max	S_max(x)
 * @param w_max		w_max(x)
 * @param w_range	range of w
 */
void computeSH(const cv::Mat& img, cv::Mat_<float>& SH_max, cv::Mat_<float>& w_max, const pair<int, int>& w_range) {
	SH_max = cv::Mat_<float>(1, img.cols, 0.0f);
	w_max = cv::Mat_<float>(1, img.cols, 0.0f);

#if 0
	ifstream in_SH("SH_max.txt");
	ifstream in_w("w_max.txt");
	if (in_SH.good() && in_w.good()) {
		in_SH.close();
		in_w.close();
		SH_max = cvutils::read("SH_max.txt");
		w_max = cvutils::read("w_max.txt");
	}
	else {
#endif
		printf("computing");
		for (int c = 0; c < img.cols; ++c) {
			printf("\rcomputing c = %d/%d  ", c, img.cols);

			cv::Mat_<float> SH(1, img.cols, 0.0f);

			for (int w = w_range.first; w <= w_range.second; ++w) {
				if (c - w < 0 || c + w >= img.cols) continue;

				cv::Mat R1 = img(cv::Rect(c, 0, w, img.rows));
				cv::Mat R2 = img(cv::Rect(c - w, 0, w, img.rows));

				SH(0, w) = MI(R1, R2);

				if (SH(0, w) > SH_max(0, c)) {
					SH_max(0, c) = SH(0, w);
					w_max(0, c) = w;
				}
			}
		}
		printf("\n");

#if 0
		// output SH_max(x) and w_max(x)
		cvutils::write("SH_max.txt", SH_max);
		cvutils::write("w_max.txt", w_max);
	}
#endif

	////////////////////////////////////////////////////////////////////////
	// DEBUG
#if 0
	{
		cv::Mat_<float> histograms(1, grayImg.cols, 0.0f);
		int x0 = 153;
		for (int c = 0; c < grayImg.cols; ++c) {
			if (abs(c - x0) < 10) continue;

			if (c < x0) {
				int w = x0 - c;
				if (x0 + w >= grayImg.cols) continue;
				cv::Mat R1 = grayImg(cv::Rect(c, 0, w, grayImg.rows - 1));
				cv::Mat R2 = grayImg(cv::Rect(c + w, 0, w, grayImg.rows - 1));
				histograms(0, c) += MI(R1, R2);
			}
			else {
				int w = c - x0;
				if (x0 - w < 0) continue;
				cv::Mat R1 = grayImg(cv::Rect(x0 - w, 0, w, grayImg.rows));
				cv::Mat R2 = grayImg(cv::Rect(x0, 0, w, grayImg.rows));
				histograms(0, c) += MI(R1, R2);
			}
		}

		cvutils::outputImageWithHorizontalGraph(img, histograms, "temp.png");

		float max_value = 0.0f;
		int max_index = -1;
		for (int i = 0; i < histograms.cols; ++i) {
			if (histograms(0, i) > max_value) {
				max_value = histograms(0, i);
				max_index = i;
			}
		}

		if (max_index < x0) {
			int w = x0 - max_index;
			cv::Mat R1 = grayImg(cv::Rect(max_index, 0, w, grayImg.rows - 1));
			cv::Mat R2 = grayImg(cv::Rect(max_index + w, 0, w, grayImg.rows - 1));
			cv::imwrite("R1.png", R1);
			cv::imwrite("R2.png", R2);
			cout << "MSD: " << cvutils::msd(R1, R2) << endl;
		}
		else {
			int w = max_index - x0;
			cv::Mat R1 = grayImg(cv::Rect(x0 - w, 0, w, grayImg.rows - 1));
			cv::Mat R2 = grayImg(cv::Rect(x0, 0, w, grayImg.rows - 1));
			cv::imwrite("R1.png", R1);
			cv::imwrite("R2.png", R2);
			cout << "MSD: " << cvutils::msd(R1, R2) << endl;
		}
	}
#endif
	////////////////////////////////////////////////////////////////////////
}

/**
 * imgから、Ver(y)とHor(x)を計算する。
 *
 * @param img		image (3-channel color image)
 * @param Ver		Ver(y)
 * @param Hor		Hor(x)
 */
void computeVerAndHor(const cv::Mat& img, cv::Mat_<float>& Ver, cv::Mat_<float>& Hor, float sigma) {
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
 *
 * @param tile		タイル画像 (3-channel image)
 * @param min_size
 * @param horizontal_edge_max		水平分割線に対する、エッジの強さの最小値
 * @return				分割する場合はtrue / false otherwise
 */
bool subdivideTile(const cv::Mat& tile, const cv::Mat& edges, int min_size, int tile_margin, cv::Rect& rect) {
	if (tile.cols < min_size || tile.rows < min_size) return false;

	//cv::imwrite("tile.png", tile);

	cv::Mat grayTile;
	cv::cvtColor(tile, grayTile, CV_BGR2GRAY);

	// sum horizontally and vertically
	cv::Mat vertical_edges;
	cv::Mat horizontal_edges;
	cv::reduce(edges, vertical_edges, 0, CV_REDUCE_SUM, CV_32F);
	cv::reduce(edges, horizontal_edges, 1, CV_REDUCE_SUM, CV_32F);

	//cvutils::outputImageWithHorizontalAndVerticalGraph(tile, horizontal_edges, vertical_edges, "tile_ver_hor.png");

	cv::Mat vertical_edges_max;
	cv::Mat horizonta_edges_max;
	cv::reduce(vertical_edges, vertical_edges_max, 1, CV_REDUCE_MAX, CV_32F);
	cv::reduce(horizontal_edges, horizonta_edges_max, 0, CV_REDUCE_MAX, CV_32F);

	float vertical_edge_threshold = tile.rows * 0.2f * 255;
	float horizontal_edge_threshold = tile.cols * 0.2f * 255;

#if 0
	for (int c = 0; c < vertical_edges.cols; ++c) {
		cout << "c: " << c << ", " << vertical_edges.at<float>(0, c) << endl;
	}
	for (int r = 0; r < horizontal_edges.rows; ++r) {
		cout << "r: " << r << ", " << horizontal_edges.at<float>(r, 0) << endl;
	}
#endif

	// find the split closest to the boundary
	int x1 = -1;
	float prev_x1;
	for (int c = tile_margin; c < vertical_edges.cols - tile_margin; ++c) {
		if (x1 == -1) {
			if (vertical_edges.at<float>(0, c) >= vertical_edge_threshold) {
				x1 = c;
				prev_x1 = vertical_edges.at<float>(0, c);
			}
		}
		else if (vertical_edges.at<float>(0, c) > prev_x1) {
			x1 = c;
			prev_x1 = vertical_edges.at<float>(0, c);
		}
		else {
			break;
		}

	}
	int x2 = -1;
	float prev_x2;
	for (int c = vertical_edges.cols - tile_margin - 1; c >= tile_margin; --c) {
		if (x2 == -1) {
			if (vertical_edges.at<float>(0, c) >= vertical_edge_threshold) {
				x2 = c;
				prev_x2 = vertical_edges.at<float>(0, c);
			}
		}
		else if (vertical_edges.at<float>(0, c) > prev_x2) {
			x2 = c;
			prev_x2 = vertical_edges.at<float>(0, c);
		}
		else {
			break;
		}
	}
	if (x1 == -1 || x2 == -1 || x1 == x2) return false;

	int y1 = -1;
	float prev_y1;
	for (int r = tile_margin; r < horizontal_edges.rows - tile_margin; ++r) {
		if (y1 == -1) {
			if (horizontal_edges.at<float>(r, 0) >= horizontal_edge_threshold) {
				y1 = r;
				prev_y1 = horizontal_edges.at<float>(r, 0);
			}
		}
		else if (horizontal_edges.at<float>(r, 0) > prev_y1) {
			y1 = r;
			prev_y1 = horizontal_edges.at<float>(r, 0);
		}
		else {
			break;
		}
	}
	int y2 = -1;
	float prev_y2;
	for (int r = horizontal_edges.rows - tile_margin - 1; r >= tile_margin; --r) {
		if (y2 == -1) {
			if (horizontal_edges.at<float>(r, 0) >= horizontal_edge_threshold) {
				y2 = r;
				prev_y2 = horizontal_edges.at<float>(r, 0);
			}
		}
		else if (horizontal_edges.at<float>(r, 0) > prev_y2) {
			y2 = r;
			prev_y2 = horizontal_edges.at<float>(r, 0);
		}
		else {
			break;
		}
	}
	if (y1 == -1 || y2 == -1 || y1 == y2) return false;


	cv::Mat result = tile.clone();
	cv::rectangle(result, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 1);
	cv::imwrite("tile_window.png", result);


	rect = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));

	return true;
}

/**
 * Ver(y)の極小値をsplit lineの候補とし、S_max(y)に基づいて最適なsplit lineの組み合わせを探す。
 *
 * @param Ver			Ver(y)
 * @param min_interval	minimum floor height
 * @param max_interval	maximum floor height
 * @param y_split		最適なsplit lineの組み合わせ
 */
void findBestHorizontalSplitLines(const cv::Mat& img, const cv::Mat_<float>& Ver, float min_interval, float max_interval, vector<int>& y_split) {
	y_split.clear();

	vector<int> y_candidates = cvutils::getPeak(Ver, false, 1, cvutils::LOCAL_MINIMUM, 1);
	y_candidates.insert(y_candidates.begin(), 0);
	y_candidates.push_back(img.rows - 1);
	
	vector<vector<float>> costs;
	vector<vector<int>> indices;
	vector<vector<int>> nums;

	// 最初の行のコストを初期化
	{
		costs.push_back(vector<float>(y_candidates.size(), numeric_limits<float>::max()));
		indices.push_back(vector<int>(y_candidates.size(), -1));
		nums.push_back(vector<int>(y_candidates.size(), 0));
		costs[0][0] = 0;
		indices[0][0] = 0;
		nums[0][0] = 0;
	}

	// 2行目以降について、Dynamic Programmingで最小コストを計算
	for (int i = 1; ; ++i) {
		costs.push_back(vector<float>(y_candidates.size(), numeric_limits<float>::max()));
		indices.push_back(vector<int>(y_candidates.size(), -1));
		nums.push_back(vector<int>(y_candidates.size(), 0));

		for (int k = 0; k < y_candidates.size() - 1; ++k) {
			bool found = false;

			for (int j = k + 1; j < y_candidates.size() - 1; ++j) {
				if (indices[i - 1][k] == -1) continue;

				if (y_candidates[j] - y_candidates[k] < min_interval) continue;
				if (found && y_candidates[j] - y_candidates[k] > max_interval) continue;

				found = true;
				float new_cost = costs[i - 1][k] + Ver(y_candidates[j], 0);
				if (new_cost / (nums[i - 1][k] + 1) < costs[i][j] / nums[i][j]) {
					costs[i][j] = new_cost;
					indices[i][j] = k;
					nums[i][j] = nums[i - 1][k] + 1;
				}
			}
		}

		for (int k = 0; k < y_candidates.size(); ++k) {
			if (indices[i - 1][k] == -1) continue;

			if (y_candidates.back() - y_candidates[k] > max_interval) continue;

			if (costs[i - 1][k] / nums[i - 1][k] < costs[i].back() / nums[i].back()) {
				costs[i].back() = costs[i - 1][k];
				indices[i].back() = k;
				nums[i].back() = nums[i - 1][k];
			}
		}

		// 最後のy以外がすべて-1なら、終了
		bool updated = false;
		for (int j = 0; j < indices[i].size() - 1; ++j) {
			if (indices[i][j] != -1) updated = true;
		}
		if (!updated) break;
	}

	// y_splitに、最適解を格納する
	y_split.push_back(y_candidates.back());
	int prev_index = y_candidates.size() - 1;
	for (int i = indices.size() - 1; i >= 1; --i) {
		int index = indices[i][prev_index];
		if (index == prev_index) continue;

		y_split.insert(y_split.begin(), y_candidates[index]);
		prev_index = index;
	}
}

/**
* Hor(x)の極小値をsplit lineの候補とし、S_max(x)に基づいて最適なsplit lineの組み合わせを探す。
*
* @param Hor			Hor(x)
* @param min_interval	minimum tile width
* @param max_interval	maximum tile width
* @param x_split		最適なsplit lineの組み合わせ
*/
void findBestVerticalSplitLines(const cv::Mat& img, const cv::Mat_<float>& Hor, float min_interval, float max_interval, vector<int>& x_split) {
	x_split.clear();

	vector<int> x_candidates = cvutils::getPeak(Hor, false, 1, cvutils::LOCAL_MINIMUM, 1);
	x_candidates.insert(x_candidates.begin(), 0);
	x_candidates.push_back(img.cols - 1);

	vector<vector<float>> costs;
	vector<vector<int>> indices;
	vector<vector<int>> nums;

	// 最初の列のコストを初期化
	{
		costs.push_back(vector<float>(x_candidates.size(), numeric_limits<float>::max()));
		indices.push_back(vector<int>(x_candidates.size(), -1));
		nums.push_back(vector<int>(x_candidates.size(), 0));
		costs[0][0] = 0;
		indices[0][0] = 0;
		nums[0][0] = 0;
	}

	// 2列目以降について、Dynamic Programmingで最小コストを計算
	for (int i = 1;; ++i) {
		costs.push_back(vector<float>(x_candidates.size(), numeric_limits<float>::max()));
		indices.push_back(vector<int>(x_candidates.size(), -1));
		nums.push_back(vector<int>(x_candidates.size(), 0));

		for (int k = 0; k < x_candidates.size() - 1; ++k) {
			bool found = false;

			for (int j = k + 1; j < x_candidates.size() - 1; ++j) {
				if (indices[i - 1][k] == -1) continue;

				if (x_candidates[j] - x_candidates[k] < min_interval) continue;
				if (found && x_candidates[j] - x_candidates[k] > max_interval) continue;

				found = true;
				float new_cost = costs[i - 1][k] + Hor(0, x_candidates[j]);
				if (new_cost / (nums[i - 1][k] + 1) < costs[i][j] / nums[i][j]) {
					costs[i][j] = new_cost;
					indices[i][j] = k;
					nums[i][j] = nums[i - 1][k] + 1;
				}
			}
		}

		for (int k = 0; k < x_candidates.size(); ++k) {
			if (indices[i - 1][k] == -1) continue;

			if (x_candidates.back() - x_candidates[k] > max_interval) continue;

			if (costs[i - 1][k] / nums[i - 1][k] < costs[i].back() / nums[i].back()) {
				costs[i].back() = costs[i - 1][k];
				indices[i].back() = k;
				nums[i].back() = nums[i - 1][k];
			}
		}

		// 最後のx以外がすべて-1なら、終了
		bool updated = false;
		for (int j = 0; j < indices[i].size() - 1; ++j) {
			if (indices[i][j] != -1) updated = true;
		}
		if (!updated) break;
	}

	// x_splitに、最適解を格納する
	x_split.push_back(x_candidates.back());
	int prev_index = x_candidates.size() - 1;
	for (int i = indices.size() - 1; i >= 1; --i) {
		int index = indices[i][prev_index];
		if (index == prev_index) continue;

		x_split.insert(x_split.begin(), x_candidates[index]);
		prev_index = index;
	}
}

void getSplitLines(const cv::Mat_<float>& mat, vector<int>& split_positions) {
	if (mat.cols == 1) {
		for (int r = 0; r < mat.rows; ++r) {
			if (cvutils::isLocalMinimum(mat, r, 1)) {
				split_positions.push_back(r);
			}
		}
	}
	else if (mat.rows == 1) {
		for (int c = 0; c < mat.cols; ++c) {
			if (cvutils::isLocalMinimum(mat, c, 1)) {
				split_positions.push_back(c);
			}
		}
	}
	
	if (split_positions.size() == 0 || split_positions[0] > 0) {
		split_positions.insert(split_positions.begin(), 0);
	}

	if (mat.cols == 1) {
		if (split_positions.back() < mat.rows - 1) {
			split_positions.push_back(mat.rows - 1);
		}
	}
	else if (mat.rows == 1) {
		if (split_positions.back() < mat.cols - 1) {
			split_positions.push_back(mat.cols - 1);
		}
	}
}

void refine(vector<int>& y_split, vector<int>& x_split, vector<vector<cv::Rect>>& window_rects) {
	// 各フロアの窓の数をカウントする
	vector<int> win_per_row(y_split.size() - 1, 0);
	int max_win_per_row = 0;
	for (int i = 0; i < y_split.size() - 1; ++i) {
		for (int j = 0; j < x_split.size() - 1; ++j) {
			if (window_rects[i][j].width > 0 && window_rects[i][j].height > 0) {
				win_per_row[i]++;
			}
		}
		if (win_per_row[i] > max_win_per_row) {
			max_win_per_row = win_per_row[i];
		}
	}

	// 各カラムの窓の数をカウントする
	vector<int> win_per_col(x_split.size() - 1, 0);
	int max_win_per_col = 0;
	for (int j = 0; j < x_split.size() - 1; ++j) {
		for (int i = 0; i < y_split.size() - 1; ++i) {
			if (window_rects[i][j].width > 0 && window_rects[i][j].height > 0) {
				win_per_col[j]++;
			}
		}
		if (win_per_col[j] > max_win_per_col) {
			max_win_per_col = win_per_col[j];
		}
	}

	// 壁のフロアかどうかチェックする
	vector<bool> is_wall_row(y_split.size() - 1, false);
	for (int i = 0; i < y_split.size() - 1; ++i) {
		if (win_per_row[i] < max_win_per_row * 0.2) {
			is_wall_row[i] = true;

			for (int j = 0; j < x_split.size() - 1; ++j) {
				window_rects[i][j] = cv::Rect(0, 0, 0, 0);
			}
		}
	}

	// 壁のカラムかどうかチェックする
	vector<bool> is_wall_col(x_split.size() - 1, false);
	for (int j = 0; j < x_split.size() - 1; ++j) {
		if (win_per_col[j] < max_win_per_col * 0.2) {
			is_wall_col[j] = true;

			for (int i = 0; i < y_split.size() - 1; ++i) {
				window_rects[i][j] = cv::Rect(0, 0, 0, 0);
			}
		}
	}

	// 窓のX座標をvoteする
	for (int j = 0; j < x_split.size() - 1; ++j) {
		if (is_wall_col[j]) continue;

		int x1, x2;

		// voteする
		vector<float> histogram1(x_split[j + 1] - x_split[j], 0);
		vector<float> histogram2(x_split[j + 1] - x_split[j], 0);
		for (int i = 0; i < y_split.size() - 1; ++i) {
			if (is_wall_row[i]) continue;
			if (window_rects[i][j].width == 0 && window_rects[i][j].height == 0) continue;

			for (int c = 0; c < histogram1.size(); ++c) {
				histogram1[c] += utils::gause(window_rects[i][j].x - c, 2);
				histogram2[c] += utils::gause(window_rects[i][j].x + window_rects[i][j].width - 1 - c, 2);
			}
		}

		// max voteを探す
		float max_val1 = 0.0f;
		float max_val2 = 0.0f;
		for (int c = 0; c < histogram1.size(); ++c) {
			if (histogram1[c] > max_val1) {
				max_val1 = histogram1[c];
				x1 = c;
			}
			if (histogram2[c] > max_val2) {
				max_val2 = histogram2[c];
				x2 = c;
			}
		}

		// 全てのフロアの窓のX座標をそろえる
		for (int r = 0; r < y_split.size() - 1; ++r) {
			if (is_wall_row[r]) continue;

			if (r == 0 || r == y_split.size() - 1) {
				int new_x = window_rects[r][j].x;
				int new_width = window_rects[r][j].width;
				if (abs(window_rects[r][j].x - x1) < 5) {
					new_x = x1;
				}
				if (abs(window_rects[r][j].x + window_rects[r][j].width - 1 - x2) < 5) {
					new_width = x2 - new_x;
				}
				window_rects[r][j].x = new_x;
				window_rects[r][j].width = new_width;
			}
			else {
				window_rects[r][j].x = x1;
				window_rects[r][j].width = x2 - x1 + 1;
			}
		}
	}

	// 窓のY座標をvoteする
	for (int i = 0; i < y_split.size() - 1; ++i) {
		if (is_wall_row[i]) continue;

		int y1, y2;

		// voteする
		vector<float> histogram1(y_split[i + 1] - y_split[i], 0);
		vector<float> histogram2(y_split[i + 1] - y_split[i], 0);
		for (int j = 0; j < x_split.size() - 1; ++j) {
			if (is_wall_col[j]) continue;
			if (window_rects[i][j].height == 0) continue;

			for (int r = 0; r < histogram1.size(); ++r) {
				histogram1[r] += utils::gause(window_rects[i][j].y - r, 2);
				histogram2[r] += utils::gause(window_rects[i][j].y + window_rects[i][j].height - 1 - r, 2);
			}
		}

		// max voteを探す
		float max_val1 = 0.0f;
		float max_val2 = 0.0f;
		for (int r = 0; r < histogram1.size(); ++r) {
			if (histogram1[r] > max_val1) {
				max_val1 = histogram1[r];
				y1 = r;
			}
			if (histogram2[r] > max_val2) {
				max_val2 = histogram2[r];
				y2 = r;
			}
		}

		// 全てのカラムの窓のY座標をそろえる
		for (int c = 0; c < x_split.size() - 1; ++c) {
			if (is_wall_col[c]) continue;

			window_rects[i][c].y = y1;
			window_rects[i][c].height = y2 - y1 + 1;
		}
	}

	// 窓のないフロアが連続している場合は、連結する
	for (int i = 0; i < is_wall_row.size() - 1;) {
		if (is_wall_row[i] && is_wall_row[i + 1]) {
			is_wall_row.erase(is_wall_row.begin() + i + 1);
			y_split.erase(y_split.begin() + i + 1);
			window_rects.erase(window_rects.begin() + i + 1);
		}
		else {
			i++;
		}
	}

	// 窓のないカラムが連続している場合は、連結する
	for (int j = 0; j < is_wall_col.size() - 1;) {
		if (is_wall_col[j] && is_wall_col[j + 1]) {
			is_wall_col.erase(is_wall_col.begin() + j + 1);
			x_split.erase(x_split.begin() + j + 1);
			for (int i = 0; i < y_split.size() - 1; ++i) {
				window_rects[i].erase(window_rects[i].begin() + j + 1);
			}
		}
		else {
			j++;
		}
	}
}

void subdivideFacade(const cv::Mat& img) {
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

	//////////////////////////////////////////////////////////////////////////////
	// obtain edge image
#if 0
	cv::Mat grad;
	cv::Mat grad_x;
	cv::Mat grad_y;
	cv::Mat abs_grad_x;
	cv::Mat abs_grad_y;
	cv::Sobel(grayImg, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	cv::Sobel(grayImg, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	//cv::imwrite("grad.png", grad);
	grad.convertTo(grayImg, CV_8U);
	//cv::imwrite("grad2.png", grayImg);
#endif
	//////////////////////////////////////////////////////////////////////////////

	// vertical split
	cv::Mat_<float> SV_max;
	cv::Mat_<float> h_max;
	computeSV(grayImg, SV_max, h_max, make_pair(3, grayImg.rows));

	// visualize SV_max(y) and h_max(y)
	outputFacadeStructureV(img, SV_max, h_max, "SV_max.png", 1);

	// estimate the floor height
	float floor_height = cvutils::getMostPopularValue(h_max, 3, 3);
	cout << "Floor height: " << floor_height << endl;
	
	// horizontal split
	cv::Mat_<float> SH_max;
	cv::Mat_<float> w_max;
	computeSH(grayImg, SH_max, w_max, make_pair(3, grayImg.cols));

	// visualize SH_max(x) and w_max(x)
	outputFacadeStructureH(img, SH_max, w_max, "SH_max.png", 1);

	float tile_width = cvutils::getMostPopularValue(w_max, 3, 3);
	cout << "Tile width: " << tile_width << endl;

	// visualize SV_max(y), h_max(y), SH_max(x) and w_max(x)
	outputFacadeStructure(img, SV_max, h_max, SH_max, w_max, "SV_SH_max.png", 1);

	// compute Ver(y) and Hor(x)
	cv::Mat_<float> Ver;
	cv::Mat_<float> Hor;
	computeVerAndHor(img, Ver, Hor, floor_height * 0.1);

	/////////////////////////////////////////////////////////////////
	// DEBUG
	cvutils::outputImageWithHorizontalAndVerticalGraph(img, Ver, Hor, "facade_subdivision_by_ver_hor.png", cvutils::LOCAL_MINIMUM, 1);
	/////////////////////////////////////////////////////////////////
	
	time_t start = clock();

	// Facadeのsplit linesを求める
	vector<int> y_split;
	//findBestHorizontalSplitLines(img, Ver, floor_height * 0.85, floor_height * 1.85, y_split);
	getSplitLines(Ver, y_split);
	vector<int> x_split;
	//findBestVerticalSplitLines(img, Hor, tile_width * 0.4, tile_width * 1.85, x_split);
	getSplitLines(Hor, x_split);
	time_t end = clock();
	cout << "Time: " << (end - start) << "msec." << endl;
	outputFacadeStructure(img, y_split, x_split, "facade_subdivision.png", 1);

	cv::Mat detected_edges;
	cv::Canny(img, detected_edges, 30, 100, 3);
	cv::imwrite("detected_edges.png", detected_edges);

	// facadeの端のエッジを削除する
	int margin = 8;
	for (int r = 0; r < detected_edges.rows; ++r) {
		for (int c = 0; c < detected_edges.cols; ++c) {
			if (r < margin || r >= detected_edges.rows - margin || c < margin || c >= detected_edges.cols - margin) {
				detected_edges.at<unsigned char>(r, c) = 0;
			}
		}
	}

#if 0
	cv::Mat detected_vertical_edges;
	cv::Mat detected_horizontal_edges;
	cv::reduce(detected_edges, detected_vertical_edges, 0, CV_REDUCE_SUM, CV_32F);
	cv::reduce(detected_edges, detected_horizontal_edges, 1, CV_REDUCE_SUM, CV_32F);

	cvutils::outputImageWithHorizontalAndVerticalGraph(detected_edges, detected_horizontal_edges, detected_vertical_edges, "detected_edges_votes.png");
	
	cv::Mat vertical_edge_max;
	cv::Mat horizontal_edge_max;
	cv::reduce(detected_vertical_edges, vertical_edge_max, 1, CV_REDUCE_MAX, CV_32F);
	cv::reduce(detected_horizontal_edges, horizontal_edge_max, 0, CV_REDUCE_MAX, CV_32F);
#endif
	
	// 各tileの窓の位置を求める
	vector<vector<cv::Rect>> window_rects;
	int window_count = 0;
	window_rects.resize(y_split.size() - 1);
	for (int i = 0; i < y_split.size() - 1; ++i) {
		window_rects[i].resize(x_split.size() - 1);
		for (int j = 0; j < x_split.size() - 1; ++j) {
			cv::Mat tile(img, cv::Rect(x_split[j], y_split[i], x_split[j + 1] - x_split[j], y_split[i + 1] - y_split[i]));
			cv::Mat tile_edges(detected_edges, cv::Rect(x_split[j], y_split[i], x_split[j + 1] - x_split[j], y_split[i + 1] - y_split[i]));
			if (subdivideTile(tile, tile_edges, 10, 1, window_rects[i][j])) {
				window_count++;
			}
			else {
				window_rects[i][j] = cv::Rect(0, 0, 0, 0);
			}
		}
	}
	cout << "Window count: " << window_count << endl;
	outputFacadeAndWindows(img, y_split, x_split, window_rects, "facade_windows.png");

	// 窓の位置をalignする
	refine(y_split, x_split, window_rects);
	outputFacadeAndWindows(img, y_split, x_split, window_rects, "facade_windows_refined.png");
}

int main() {
	cv::Mat img = cv::imread("../testdata/50.png");
	//cv::Mat img = cv::imread("\\\\matrix.cs.purdue.edu\\cgvlab\\gen\\meeting\\2016\\20160531\\facade_images\\facade4.png");

	subdivideFacade(img);

	return 0;
}
