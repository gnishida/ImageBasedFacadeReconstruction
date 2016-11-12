#include "CVUtils.h"
#include <fstream>
#include <iostream>
#include <regex>
#include "Utils.h"

#ifndef SQR
#define SQR(x)	((x) * (x))
#endif

using namespace std;

namespace cvutils {

	/**
	 * Find the maximum element value of the matrix and return it.
	 *
	 * @param mat	Matrix
	 * @return		Maximum element value
	 */
	double max(const cv::Mat& mat) {
		cv::Mat result = mat.clone();
		result = result.reshape(1, result.cols * result.rows * result.channels());
		cv::reduce(result, result, 0, CV_REDUCE_MAX);
		result.convertTo(result, CV_64F);

		return result.at<double>(0, 0);
	}

	/**
	* Find the minimum element value of the matrix and return it.
	*
	* @param mat	Matrix
	* @return		Minimum element value
	*/
	double min(const cv::Mat& mat) {
		cv::Mat result = mat.clone();
		result = result.reshape(1, result.cols * result.rows * result.channels());
		cv::reduce(result, result, 0, CV_REDUCE_MIN);
		result.convertTo(result, CV_64F);
		
		return result.at<double>(0, 0);
	}

	vector<double> mean(const cv::Mat& mat) {
		cv::Mat result = mat.clone();
		result = result.reshape(1, result.cols * result.rows);
		cv::reduce(result, result, 0, CV_REDUCE_AVG);
		result.convertTo(result, CV_64F);

		vector<double> ret(mat.channels());
		for (int i = 0; i < ret.size(); ++i) {
			ret[i] = result.at<double>(0, i);
		}
		return ret;
	}

	/**
	 * Write out the matrix to the file.
	 * Each line corresponds to a row, and each element is delimitered by '\t'.
	 *
	 * @param filename		Output file name
	 * @param mat			Matrix
	 */
	void write(const string& filename, const cv::Mat& mat) {
		ofstream out(filename);

		for (int r = 0; r < mat.rows; ++r) {
			for (int c = 0; c < mat.cols; ++c) {
				if (c > 0) out << "\t";

				out << get(mat, r, c);
			}
			out << endl;
		}

		out.close();
	}

	/**
	 * Read the matrix from the file.
	 * Each line corresponds to a row, and each element is delimitered by '\t'.
	 * Note: The float type is assumed for the element of the matrix.
	 *
	 * @param filename		Input file name
	 * @return				Matrix
	 */
	cv::Mat read(const string& filename) {
		ifstream in(filename);
		vector<vector<float>> data;

		regex re("[\\s,]+");

		while (!in.eof()) {
			string str;
			getline(in, str);

			if (str.empty()) break;

			data.push_back(vector<float>());

			sregex_token_iterator it(str.begin(), str.end(), re, -1);
			sregex_token_iterator reg_end;
			for (; it != reg_end; ++it) {
				data.back().push_back(stof(it->str()));
			}
		}

		cv::Mat result = cv::Mat(data.size(), data[0].size(), CV_32F);

		for (int r = 0; r < data.size(); ++r) {
			for (int c = 0; c < data[r].size(); ++c) {
				result.at<float>(r, c) = data[r][c];
			}
		}

		return result;
	}

	/**
	 * Return the mean squared differences between two images.
	 *
	 * @param img1		Image 1 (1-channel image)
	 * @param img2		Image 2 (1-channel image)
	 * @return			Mean squared differences
	 */
	double msd(const cv::Mat& img1, const cv::Mat& img2) {
		cv::Mat diff_mat;
		cv::absdiff(img1, img2, diff_mat);
		diff_mat = diff_mat.reshape(1, diff_mat.rows * diff_mat.cols * diff_mat.channels());

		cv::Mat sqr_mat;
		cv::multiply(diff_mat, diff_mat, sqr_mat, 1, CV_64F);

		cv::reduce(sqr_mat, sqr_mat, 0, CV_REDUCE_SUM);
		return sqr_mat.at<double>(0, 0) / img1.rows / img1.cols;
	}

	double corr(const cv::Mat& img1, const cv::Mat& img2) {
		cv::Mat a;
		img1.convertTo(a, CV_64F);
		cv::Mat b;
		img2.convertTo(b, CV_64F);

		cv::Mat m1, m2, m3;
		cv::multiply(a, b, m1);
		cv::multiply(a, a, m2);
		cv::multiply(b, b, m3);

		m1 = m1.reshape(1, m1.rows * m1.cols * m1.channels());
		m2 = m2.reshape(1, m2.rows * m2.cols * m2.channels());
		m3 = m3.reshape(1, m3.rows * m3.cols * m3.channels());
		m1.convertTo(m1, CV_64F);
		m2.convertTo(m2, CV_64F);
		m3.convertTo(m3, CV_64F);
		cv::reduce(m1, m1, 0, CV_REDUCE_SUM, CV_64F);
		cv::reduce(m2, m2, 0, CV_REDUCE_SUM, CV_64F);
		cv::reduce(m3, m3, 0, CV_REDUCE_SUM, CV_64F);

		return m1.at<double>(0, 0) / sqrt(m2.at<double>(0, 0) * m3.at<double>(0, 0));
	}

	double get(const cv::Mat& mat, int r, int c) {
		if (mat.type() == CV_8U) {
			return mat.at<unsigned char>(r, c);
		}
		else if (mat.type() == CV_32F) {
			return mat.at<float>(r, c);
		}
		else if (mat.type() == CV_64F) {
			return mat.at<double>(r, c);
		}
		else {
			return 0.0;
		}
	}

	void grayScale(const cv::Mat& img, cv::Mat& grayImg) {
		if (img.channels() == 1) {
			grayImg = img.clone();
		}
		else if (img.channels() >= 3) {
			std::vector<cv::Mat> img_channels;
			cv::split(img, img_channels);
			grayImg = img_channels[0] * 0.114 + img_channels[1] * 0.587 + img_channels[2] * 0.299;
		}
	}

	void blend(cv::Mat foreground, cv::Mat background, cv::Mat& blended) {
		// if there is no alpha channel in the foregound, copy the foreground to the result.
		if (foreground.channels() < 4) {
			blended = foreground.clone();
			return;
		}

		// convert the background to 3-channel image
		if (background.channels() == 1) {
			cv::cvtColor(background, background, cv::COLOR_GRAY2BGR);
		}
		else if (background.channels() == 4) {
			cv::cvtColor(background, background, cv::COLOR_BGRA2BGR);
		}

		blended = background.clone();

		for (int r = 0; r < blended.rows; ++r) {
			for (int c = 0; c < blended.cols; ++c) {
				cv::Vec3b c1 = background.at<cv::Vec3b>(r, c);
				cv::Vec4b c2 = foreground.at<cv::Vec4b>(r, c);
					
				float alpha = (float)c2[3] / 255;
				for (int k = 0; k < 3; ++k) {
					c1[k] = c1[k] * (1 - alpha) + c2[k] * alpha;
				}

				blended.at<cv::Vec3b>(r, c) = c1;
			}
		}
	}

	void grad(cv::Mat img, cv::Mat& grad) {
		cv::Mat grayImg;
		grayScale(img, grayImg);

		cv::Mat sobelx;
		cv::Mat sobely;
		cv::Sobel(grayImg, sobelx, CV_32F, 1, 0);
		cv::Sobel(grayImg, sobely, CV_32F, 0, 1);

		grad = cv::Mat(grayImg.size(), CV_32F);
		for (int r = 0; r < grayImg.rows; ++r) {
			for (int c = 0; c < grayImg.cols; ++c) {
				grad.at<float>(r, c) = std::hypot(sobelx.at<float>(r, c), sobely.at<float>(r, c));
			}
		}
	}

	void median(cv::Mat img, int x, int y, cv::Mat& med) {
		if (x == 1 && y == 0) {
			med = cv::Mat(img.rows, 1, CV_32F);
			for (int r = 0; r < img.rows; ++r) {
				std::map<double, int> hist;
				for (int c = 0; c < img.cols; ++c) {
					double val = get(img, r, c);
					if (hist.find(val) == hist.end()) {
						hist[val] = 0;
					}
					hist[val]++;
				}
				
				int cnt = 0;
				for (auto it = hist.begin(); it != hist.end(); ++it, ++cnt) {
					if (cnt >= img.cols / 2) {
						med.at<float>(r, 0) = it->first;
						break;
					}
				}
			}
		}
		else {
			med = cv::Mat(1, img.cols, CV_32F);
			for (int c = 0; c < img.cols; ++c) {
				std::map<double, int> hist;
				for (int r = 0; r < img.rows; ++r) {
					double val = get(img, r, c);
					if (hist.find(val) == hist.end()) {
						hist[val] = 0;
					}
					hist[val]++;
				}

				int cnt = 0;
				for (auto it = hist.begin(); it != hist.end(); ++it, ++cnt) {
					if (cnt >= img.rows / 2) {
						med.at<float>(0, c) = it->first;
						break;
					}
				}
			}
		}
	}

	bool isLocalMinimum(const cv::Mat& mat, int index, int num) {
		bool localMinimum = true;

		if (mat.rows == 1) {	// row vector
			if (index == 0) {
				localMinimum = false;
			}
			else if (index == mat.cols - 1) {
				localMinimum = false;
			}
			else {
				for (int c = std::max(0, index - num); c < std::min(mat.cols - 1, index + num); ++c) {
					if (get(mat, 0, c) < get(mat, 0, index)) {
						localMinimum = false;
						break;
					}
				}
			}
		}
		else if (mat.cols == 1) {	// column vector
			if (index == 0) {
				localMinimum = false;
			}
			else if (index == mat.rows - 1) {
				localMinimum = false;
			}
			else {
				for (int r = std::max(0, index - num); r < std::min(mat.rows - 1, index + num); ++r) {
					if (get(mat, r, 0) < get(mat, index, 0)) {
						localMinimum = false;
						break;
					}
				}
			}
		}

		return localMinimum;
	}

	bool isLocalMaximum(const cv::Mat& mat, int index, int num) {
		bool localMaximum = true;

		if (mat.rows == 1) {	// row vector
			if (index == 0) {
				localMaximum = false;
			}
			else {
				for (int c = index - 1; c >= std::max(0, index - num); --c) {
					if (get(mat, 0, c) > get(mat, 0, index)) {
						localMaximum = false;
						break;
					}
				}
			}

			if (localMaximum) {
				if (index == mat.cols - 1) {
					localMaximum = false;
				}
				else {
					for (int c = index + 1; c <= std::min(mat.cols - 1, index + num); ++c) {
						if (get(mat, 0, c) > get(mat, 0, index)) {
							localMaximum = false;
							break;
						}
					}
				}
			}
		}
		else if (mat.cols == 1) {	// column vector
			if (index == 0) {
				localMaximum = false;
			}
			else {
				for (int r = index - 1; r >= std::max(0, index - num); --r) {
					if (get(mat, r, 0) > get(mat, index, 0)) {
						localMaximum = false;
						break;
					}
				}
			}

			if (localMaximum) {
				if (index == mat.rows - 1) {
					localMaximum = false;
				}
				else {
					for (int r = index + 1; r <= std::min(mat.rows - 1, index + num); ++r) {
						if (get(mat, r, 0) > get(mat, index, 0)) {
							localMaximum = false;
							break;
						}
					}
				}
			}
		}

		return localMaximum;
	}

	/**
	 * 極値となるインデックスのリストを返却する。
	 *
	 * @param mat				データ
	 * @param smooth			スムースするか
	 * @param sigma				スムースする際のsigma
	 * @param flag
	 * @param continuous_num	極値周辺で、チェックする幅
	 * @return					極値となるインデックスのリスト
	 */
	vector<int> getPeak(const cv::Mat& mat, bool smooth, int sigma, int flag, int continuous_num) {
		vector<int> peaks;

		// sigma has to be odd.
		if (sigma % 2 == 0) sigma++;

		cv::Mat mat_copy;
		if (smooth) {
			cv::GaussianBlur(mat, mat_copy, cv::Size(sigma, sigma), 0, 0);
		}
		else {
			mat_copy = mat.clone();
		}

		if (mat.cols == 1) {
			for (int r = 0; r < mat.rows; ++r) {
				if (flag == LOCAL_MINIMUM) {
					if (cvutils::isLocalMinimum(mat, r, continuous_num)) {
						peaks.push_back(r);
					}
				}
				else if (flag == LOCAL_MAXIMUM) {
					if (cvutils::isLocalMaximum(mat, r, continuous_num)) {
						peaks.push_back(r);
					}

				}
			}
		}
		else if (mat.rows == 1) {
			for (int c = 0; c < mat.cols; ++c) {
				if (flag == LOCAL_MINIMUM) {
					if (cvutils::isLocalMinimum(mat, c, continuous_num)) {
						peaks.push_back(c);
					}
				}
				else if (flag == LOCAL_MAXIMUM) {
					if (cvutils::isLocalMaximum(mat, c, continuous_num)) {
						peaks.push_back(c);
					}
				}
			}
		}

		return peaks;
	}

	/**
	* matの最も頻度の多い値を返却する。
	*/
	float getMostPopularValue(const cv::Mat& h_max, float sigma, float min_value) {
		vector<float> histogram((int)cvutils::max(h_max), 0.0f);

		for (int r = 0; r < h_max.rows; ++r) {
			for (int c = 0; c < h_max.cols; ++c) {
				if (get(h_max, r, c) < min_value) continue;

				for (int i = 0; i < histogram.size(); ++i) {
					histogram[i] += utils::gause(get(h_max, r, c) - i, sigma);
				}
			}
		}

		float max_value = 0.0f;
		int max_index = -1;
		for (int i = 0; i < histogram.size(); ++i) {
			if (histogram[i] > max_value) {
				max_value = histogram[i];
				max_index = i;
			}
		}

		return max_index;
	}

	/**
	* 指定されたindexから周辺の値を調べ、極大値を返却する。
	*/
	float findNextMax(cv::Mat mat, int index, int& max_index) {
		if (mat.rows == 1) {
			mat = mat.t();
		}

		bool foundLocalMin = false;
		float val = mat.at<float>(index, 0);
		for (int r = index - 1; r >= 0; --r) {
			if (!foundLocalMin) {
				if (mat.at<float>(r, 0) > val) {
					foundLocalMin = true;
				}
				val = mat.at<float>(r, 0);
			}
			else {
				if (mat.at<float>(r, 0) > val) {
					val = mat.at<float>(r, 0);
				}
				else {
					break;
				}
			}
		}
		float max_val = val;

		foundLocalMin = false;
		val = mat.at<float>(index, 0);
		for (int r = index + 1; r < mat.rows; ++r) {
			if (!foundLocalMin) {
				if (mat.at<float>(r, 0) > val) {
					foundLocalMin = true;
				}
				val = mat.at<float>(r, 0);
			}
			else {
				if (mat.at<float>(r, 0) > val) {
					val = mat.at<float>(r, 0);
				}
				else {
					break;
				}
			}
		}

		if (val > max_val) max_val = val;

		return max_val;
	}

	/**
	* 指定されたindexから周辺の値を調べ、極大値を返却する。
	*/
	bool findNextMax(cv::Mat mat, int index, int dir, int& max_index, float& max_value) {
		if (mat.rows == 1) {
			mat = mat.t();
		}

		bool foundLocalMin = false;
		float val = mat.at<float>(index, 0);
		if (dir == -1) {
			for (int r = index - 1; r >= 0; --r) {
				if (!foundLocalMin) {
					if (mat.at<float>(r, 0) > val) {
						foundLocalMin = true;
					}
					val = mat.at<float>(r, 0);
				}
				else {
					if (mat.at<float>(r, 0) > val) {
						val = mat.at<float>(r, 0);
					}
					else {
						max_index = r + 1;
						max_value = val;
						return true;
					}
				}
			}
			return false;
		}
		else {
			for (int r = index + 1; r < mat.rows; ++r) {
				if (!foundLocalMin) {
					if (mat.at<float>(r, 0) > val) {
						foundLocalMin = true;
					}
					val = mat.at<float>(r, 0);
				}
				else {
					if (mat.at<float>(r, 0) > val) {
						val = mat.at<float>(r, 0);
					}
					else {
						max_index = r - 1;
						max_value = val;
						return true;
					}
				}
			}
			return false;
		}
	}

	/**
	* Output an image with vertical graph.
	*
	* @param img		Image
	* @param ver		Vertical graph (Nx1 matrix)
	* @param filename	output file name
	*/
	void outputImageWithVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const string& filename, int flag, int continuous_num, int lineWidth) {
		int graphSize = std::max(100.0, std::max(img.rows, img.cols) * 0.3);
		int margin = 10;

		cv::Mat result;
		cv::Scalar graph_color;
		cv::Scalar peak_color;
		if (img.type() == CV_8U) {
			graph_color = cv::Scalar(0);
			peak_color = cv::Scalar(128);
			result = cv::Mat(img.rows, img.cols + graphSize + margin + 3, img.type(), cv::Scalar(255));
		}
		else if (img.type() == CV_8UC3) {
			graph_color = cv::Scalar(0, 0, 0);
			peak_color = cv::Scalar(0, 0, 255);
			result = cv::Mat(img.rows, img.cols + graphSize + margin + 3, img.type(), cv::Scalar(255, 255, 255));
		}

		// copy img to result
		cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
		img.copyTo(roi);

		// get the maximum value of Ver(y)
		float max_ver = cvutils::max(ver);
		float min_ver = cvutils::min(ver);

		// draw vertical graph
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + margin + (get(ver, r, 0) - min_ver) / (max_ver - min_ver) * graphSize;
			int x2 = img.cols + margin + (get(ver, r + 1, 0) - min_ver) / (max_ver - min_ver) * graphSize;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), graph_color, 1, cv::LINE_8);

			if (flag & LOCAL_MINIMUM) {
				if (isLocalMinimum(ver, r, continuous_num)) {
					cv::line(result, cv::Point(0, r), cv::Point(img.cols - 1, r), peak_color, lineWidth);
				}
			}
			if (flag & LOCAL_MAXIMUM) {
				if (isLocalMaximum(ver, r, continuous_num)) {
					cv::line(result, cv::Point(0, r), cv::Point(img.cols - 1, r), peak_color, lineWidth);
				}
			}
		}

		cv::imwrite(filename, result);
	}

	/**
	* Output an image with horizontal graph.
	*
	* @param img		Image
	* @param ver		Horizontal graph (1xN matrix)
	* @param filename	output file name
	*/
	void outputImageWithHorizontalGraph(const cv::Mat& img, const cv::Mat& hor, const string& filename, int flag, int continuous_num, int lineWidth) {
		int graphSize = std::max(100.0, std::max(img.rows, img.cols) * 0.3);
		int margin = 10;

		cv::Mat result;
		cv::Scalar graph_color;
		cv::Scalar peak_color;
		if (img.type() == CV_8U) {
			graph_color = cv::Scalar(0);
			peak_color = cv::Scalar(128);
			result = cv::Mat(img.rows + graphSize + margin + 3, img.cols, img.type(), cv::Scalar(255));
		}
		else if (img.type() == CV_8UC3) {
			graph_color = cv::Scalar(0, 0, 0);
			peak_color = cv::Scalar(0, 0, 255);
			result = cv::Mat(img.rows + graphSize + margin + 3, img.cols, img.type(), cv::Scalar(255, 255, 255));
		}

		// copy img to result
		cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
		img.copyTo(roi);

		// get the maximum value of Hor(x)
		float max_hor = cvutils::max(hor);
		float min_hor = cvutils::min(hor);

		// draw horizontal graph
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + margin + (get(hor, 0, c) - min_hor) / (max_hor - min_hor) * graphSize;
			int y2 = img.rows + margin + (get(hor, 0, c + 1) - min_hor) / (max_hor - min_hor) * graphSize;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), graph_color, 1, cv::LINE_8);

			if (flag & LOCAL_MINIMUM) {
				if (isLocalMinimum(hor, c, continuous_num)) {
					cv::line(result, cv::Point(c, 0), cv::Point(c, img.rows - 1), peak_color, lineWidth);
				}
			}
			if (flag & LOCAL_MAXIMUM) {
				if (isLocalMaximum(hor, c, continuous_num)) {
					cv::line(result, cv::Point(c, 0), cv::Point(c, img.rows - 1), peak_color, lineWidth);
				}
			}
		}

		cv::imwrite(filename, result);
	}

	/**
	 * Output an image with vertical and horizontal graphs.
	 *
	 * @param img		Image
	 * @param ver		Vertical graph (Nx1 matrix)
	 * @param hor		Horizontal graph (1xN matrix)
	 * @param filename	output file name
	 */
	void outputImageWithHorizontalAndVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const cv::Mat& hor, const string& filename, int flag, int continuous_num, int lineWidth) {
		int graphSize = std::max(10.0, std::max(img.rows, img.cols) * 0.3);

		cv::Mat result;
		cv::Scalar graph_color;
		cv::Scalar peak_color;
		if (img.type() == CV_8U) {
			result = cv::Mat(img.rows + graphSize + 3, img.cols + graphSize + 3, img.type(), cv::Scalar(255));
			graph_color = cv::Scalar(0);
			peak_color = cv::Scalar(128);
		}
		else if (img.type() == CV_8UC3) {
			result = cv::Mat(img.rows + graphSize + 3, img.cols + graphSize + 3, img.type(), cv::Scalar(255, 255, 255));
			graph_color = cv::Scalar(0, 0, 0);
			peak_color = cv::Scalar(0, 0, 255);
		}

		// copy img to result
		cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
		img.copyTo(roi);
		
		// get the maximum value of Ver(y) and Hor(x)
		float max_ver = cvutils::max(ver);
		float min_ver = cvutils::min(ver);
		float max_hor = cvutils::max(hor);
		float min_hor = cvutils::min(hor);

		// draw vertical graph
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + (get(ver, r, 0) - min_ver) / (max_ver - min_ver) * graphSize;
			int x2 = img.cols + (get(ver, r + 1, 0) - min_ver) / (max_ver - min_ver) * graphSize;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), graph_color, 1, cv::LINE_8);

			if (flag & LOCAL_MINIMUM) {
				if (isLocalMinimum(ver, r, continuous_num)) {
					cv::line(result, cv::Point(0, r), cv::Point(img.cols - 1, r), peak_color, lineWidth);
				}
			}
			if (flag & LOCAL_MAXIMUM) {
				if (isLocalMaximum(ver, r, continuous_num)) {
					cv::line(result, cv::Point(0, r), cv::Point(img.cols - 1, r), peak_color, lineWidth);
				}
			}
		}

		// draw horizontal graph
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + (get(hor, 0, c) - min_hor) / (max_hor - min_hor) * graphSize;
			int y2 = img.rows + (get(hor, 0, c + 1) - min_hor) / (max_hor - min_hor) * graphSize;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), graph_color, 1, cv::LINE_8);

			if (flag & LOCAL_MINIMUM) {
				if (isLocalMinimum(hor, c, continuous_num)) {
					cv::line(result, cv::Point(c, 0), cv::Point(c, img.rows - 1), peak_color, lineWidth);
				}
			}
			if (flag & LOCAL_MAXIMUM) {
				if (isLocalMaximum(hor, c, continuous_num)) {
					cv::line(result, cv::Point(c, 0), cv::Point(c, img.rows - 1), peak_color, lineWidth);
				}
			}
		}

		cv::imwrite(filename, result);
	}

	float computeImageDist(const cv::Mat& img1, const cv::Mat& img2) {
		cv::Mat img;
		cv::resize(img1, img, cv::Size(img2.cols, img2.rows));
		return cvutils::msd(img2, img) + SQR(img1.cols - img2.cols) + SQR(img1.rows - img2.rows);
	}

	void clusterImages(const vector<cv::Mat>& images, vector<int>& labels, vector<cv::Mat>& centers, int max_centers) {
		float threshold = 2000.0f;

		while (true) {
			centers.clear();
			labels.resize(images.size());
			vector<vector<int>> clusters;

			for (int i = 0; i < images.size(); ++i) {
				float min_dist = numeric_limits<float>::max();
				int min_id;
				for (int k = 0; k < centers.size(); ++k) {
					float dist = computeImageDist(images[i], centers[k]);
					if (dist < min_dist) {
						min_dist = dist;
						min_id = k;
					}
				}

				if (min_dist < threshold) {
					labels[i] = min_id;
					clusters[min_id].push_back(i);
					int width_total = 0;
					int height_total = 0;
					for (int k = 0; k < clusters[min_id].size(); ++k) {
						width_total += images[clusters[min_id][k]].cols;
						height_total += images[clusters[min_id][k]].rows;
					}
					int width = width_total / clusters[min_id].size();
					int height = height_total / clusters[min_id].size();
					centers[min_id] = cv::Mat(height, width, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
					for (int k = 0; k < clusters[min_id].size(); ++k) {
						cv::Mat img;
						cv::resize(images[clusters[min_id][k]], img, cv::Size(centers[min_id].cols, centers[min_id].rows));
						img.convertTo(img, CV_32FC3);
						centers[min_id] += img;
					}
					centers[min_id] /= clusters[min_id].size();
					centers[min_id].convertTo(centers[min_id], CV_8UC3);
				}
				else {
					centers.push_back(images[i].clone());
					clusters.push_back(vector<int>());
					clusters.back().push_back(i);
					labels[i] = centers.size() - 1;
				}
			}

			if (centers.size() <= max_centers) break;

			threshold *= 1.2;
		}
	}
}