#include "visualization.h"
#include "CVUtils.h"

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
 * Facadeのフロアをクラスタリング結果に基づいて塗り分けする。
 *
 * @param img		Facade画像
 * @param y_split	フロアの境界 (y座標(画像の上端がy=0)が昇順に格納されている)
 * @param floors	クラスタリング結果 (1Fから順番に格納されている)
 * @param filename	ファイル名
 */
void outputFacadeSegmentation(const cv::Mat& img, const vector<int>& y_split, const vector<int>& labels, const string& filename) {
	map<int, bool> label_set;
	for (int i = 0; i < labels.size(); ++i) {
		label_set[labels[i]] = true;
	}

	// 原色パレットの作成
	vector<cv::Scalar> predefined_colors(13);
	predefined_colors[0] = cv::Scalar(255, 0, 0);
	predefined_colors[1] = cv::Scalar(0, 0, 255);
	predefined_colors[2] = cv::Scalar(0, 255, 0);
	predefined_colors[3] = cv::Scalar(255, 0, 255);
	predefined_colors[4] = cv::Scalar(0, 255, 255);
	predefined_colors[5] = cv::Scalar(255, 255, 0);
	predefined_colors[6] = cv::Scalar(128, 128, 128);
	predefined_colors[7] = cv::Scalar(128, 0, 0);
	predefined_colors[8] = cv::Scalar(0, 0, 128);
	predefined_colors[9] = cv::Scalar(0, 128, 0);
	predefined_colors[10] = cv::Scalar(128, 0, 128);
	predefined_colors[11] = cv::Scalar(0, 128, 128);
	predefined_colors[12] = cv::Scalar(128, 128, 0);

	// カラーパレットの作成
	map<int, cv::Scalar> colors;
	if (label_set.size() <= 13) {
		int count = 0;
		for (auto it = label_set.begin(); it != label_set.end(); ++it, ++count) {
			colors[it->first] = predefined_colors[count];
		}
	}
	else {
		for (auto it = label_set.begin(); it != label_set.end(); ++it) {
			colors[it->first] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
		}
	}

	cv::Mat result(img.rows, img.cols, CV_8UC3);
	for (int i = 0; i < labels.size(); ++i) {
		int y = y_split[y_split.size() - i - 2];
		int h = y_split[y_split.size() - i - 1] - y_split[y_split.size() - i - 2];
		cv::rectangle(result, cv::Rect(0, y, img.cols, h), colors[labels[i]], -1);
	}
	
	cv::imwrite(filename.c_str(), result);
}

void outputFloorSegmentation(const cv::Mat& img, const vector<int>& y_split, const vector<int>& x_split, const vector<vector<int>>& labels, const string& filename) {
	map<int, bool> label_set;
	for (int i = 0; i < labels.size(); ++i) {
		for (int j = 0; j < labels[i].size(); ++j) {
			label_set[labels[i][j]] = true;
		}
	}

	// 原色パレットの作成
	vector<cv::Scalar> predefined_colors(13);
	predefined_colors[0] = cv::Scalar(255, 0, 0);
	predefined_colors[1] = cv::Scalar(0, 0, 255);
	predefined_colors[2] = cv::Scalar(0, 255, 0);
	predefined_colors[3] = cv::Scalar(255, 0, 255);
	predefined_colors[4] = cv::Scalar(0, 255, 255);
	predefined_colors[5] = cv::Scalar(255, 255, 0);
	predefined_colors[6] = cv::Scalar(128, 128, 128);
	predefined_colors[7] = cv::Scalar(128, 0, 0);
	predefined_colors[8] = cv::Scalar(0, 0, 128);
	predefined_colors[9] = cv::Scalar(0, 128, 0);
	predefined_colors[10] = cv::Scalar(128, 0, 128);
	predefined_colors[11] = cv::Scalar(0, 128, 128);
	predefined_colors[12] = cv::Scalar(128, 128, 0);

	// カラーパレットの作成
	map<int, cv::Scalar> colors;
	if (label_set.size() <= 13) {
		int count = 0;
		for (auto it = label_set.begin(); it != label_set.end(); ++it, ++count) {
			colors[it->first] = predefined_colors[count];
		}
	}
	else {
		for (auto it = label_set.begin(); it != label_set.end(); ++it) {
			colors[it->first] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
		}
	}

	cv::Mat result(img.rows, img.cols, CV_8UC3);
	for (int i = 0; i < labels.size(); ++i) {
		int y = y_split[y_split.size() - i - 2];
		int h = y_split[y_split.size() - i - 1] - y_split[y_split.size() - i - 2];

		for (int j = 0; j < labels[i].size(); ++j) {
			int x = x_split[j];
			int w = x_split[j + 1] - x_split[j];

			cv::rectangle(result, cv::Rect(x, y, w, h), colors[labels[i][j]], -1);
		}
	}

	cv::imwrite(filename.c_str(), result);
}