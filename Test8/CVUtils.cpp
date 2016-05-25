#include "CVUtils.h"
#include <fstream>
#include <iostream>
#include <regex>
#include "Utils.h"

using namespace std;

namespace cvutils {

	/**
	 * Find the maximum element value of the matrix and return it.
	 *
	 * @param mat	Matrix
	 * @return		Maximum element value
	 */
	float max(const cv::Mat& mat) {
		cv::Mat result;

		cv::reduce(mat, result, 0, CV_REDUCE_MAX);
		cv::reduce(result, result, 1, CV_REDUCE_MAX);

		return result.at<float>(0, 0);
	}

	/**
	* Find the minimum element value of the matrix and return it.
	*
	* @param mat	Matrix
	* @return		Minimum element value
	*/
	float min(const cv::Mat& mat) {
		cv::Mat result;

		cv::reduce(mat, result, 0, CV_REDUCE_MIN);
		cv::reduce(result, result, 1, CV_REDUCE_MIN);

		return result.at<float>(0, 0);
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

				if (mat.type() == CV_8U) {
					out << mat.at<unsigned int>(r, c);
				}
				else if (mat.type() == CV_32F) {
					out << mat.at<float>(r, c);
				}
				else if (mat.type() == CV_64F) {
					out << mat.at<double>(r, c);
				}
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
		double result = 0.0f;

		for (int r = 0; r < img1.rows; ++r) {
			for (int c = 0; c < img1.cols; ++c) {
				int c1 = img1.at<unsigned char>(r, c);
				int c2 = img2.at<unsigned char>(r, c);
				int diff = img1.at<unsigned char>(r, c) - img2.at<unsigned char>(r, c);
				result += (double)(diff * diff) / img1.rows / img1.cols;
			}
		}

		return result;
	}

	bool localMinimum(const cv::Mat& mat, int index, int num) {
		bool localMinimum = true;

		if (mat.rows == 1) {	// row vector
			if (index == 0) {
				localMinimum = false;
			}
			else {
				for (int c = index; c > std::max(0, index - num); --c) {
					if (mat.at<float>(0, c) >= mat.at<float>(0, c - 1)) {
						localMinimum = false;
						break;
					}
				}
			}

			if (localMinimum) {
				if (index == mat.cols - 1) {
					localMinimum = false;
				}
				else {
					for (int c = index; c < std::min(mat.cols - 1, index + num); ++c) {
						if (mat.at<float>(0, c) >= mat.at<float>(0, c + 1)) {
							localMinimum = false;
							break;
						}
					}
				}
			}
		}
		else if (mat.cols == 1) {	// column vector
			if (index == 0) {
				localMinimum = false;
			}
			else {
				for (int r = index; r > std::max(0, index - num); --r) {
					if (mat.at<float>(r, 0) >= mat.at<float>(r - 1, 0)) {
						localMinimum = false;
						break;
					}
				}
			}

			if (localMinimum) {
				if (index == mat.rows - 1) {
					localMinimum = false;
				}
				else {
					for (int r = index; r < std::min(mat.rows - 1, index + num); ++r) {
						if (mat.at<float>(r, 0) >= mat.at<float>(r + 1, 0)) {
							localMinimum = false;
							break;
						}
					}
				}
			}
		}

		return localMinimum;
	}

	/**
	* matの最も頻度の多い値を返却する。
	*/
	float getMostPopularValue(const cv::Mat& h_max, float sigma, float min_value) {
		vector<float> histogram((int)cvutils::max(h_max), 0.0f);

		for (int r = 0; r < h_max.rows; ++r) {
			for (int c = 0; c < h_max.cols; ++c) {
				if (h_max.at<float>(r, c) < min_value) continue;

				for (int i = 0; i < histogram.size(); ++i) {
					histogram[i] += utils::gause(h_max.at<float>(r, c) - i, sigma);
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
	* Output an image with vertical graph.
	*
	* @param img		Image (3-channel color image)
	* @param ver		Vertical graph (Nx1 float-element matrix)
	* @param filename	output file name
	*/
	void outputImageWithVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const string& filename) {
		int graphSize = std::max(100.0, std::max(img.rows, img.cols) * 0.3);
		int margin = 10;

		cv::Mat result(img.rows, img.cols + graphSize + margin + 3, CV_8UC3, cv::Scalar(255, 255, 255));

		// copy img to result
		cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
		img.copyTo(roi);

		// get the maximum value of Ver(y)
		float max_ver = cvutils::max(ver);
		float min_ver = cvutils::min(ver);

		// draw vertical graph
		for (int r = 0; r < img.rows - 1; ++r) {
			int x1 = img.cols + margin + (ver.at<float>(r, 0) - min_ver) / (max_ver - min_ver) * graphSize;
			int x2 = img.cols + margin + (ver.at<float>(r + 1, 0) - min_ver) / (max_ver - min_ver) * graphSize;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		cv::imwrite(filename, result);
	}

	/**
	* Output an image with horizontal graph.
	*
	* @param img		Image (3-channel color image)
	* @param ver		Horizontal graph (1xN float-element matrix)
	* @param filename	output file name
	*/
	void outputImageWithHorizontalGraph(const cv::Mat& img, const cv::Mat& hor, const string& filename) {
		int graphSize = std::max(100.0, std::max(img.rows, img.cols) * 0.3);
		int margin = 10;

		cv::Mat result(img.rows + graphSize + margin + 3, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));

		// copy img to result
		cv::Mat roi(result, cv::Rect(0, 0, img.cols, img.rows));
		img.copyTo(roi);

		// get the maximum value of Hor(x)
		float max_hor = cvutils::max(hor);
		float min_hor = cvutils::min(hor);

		// draw horizontal graph
		for (int c = 0; c < img.cols - 1; ++c) {
			int y1 = img.rows + margin + (hor.at<float>(0, c) - min_hor) / (max_hor - min_hor) * graphSize;
			int y2 = img.rows + margin + (hor.at<float>(0, c + 1) - min_hor) / (max_hor - min_hor) * graphSize;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);
		}

		cv::imwrite(filename, result);
	}

	/**
	 * Output an image with vertical and horizontal graphs.
	 *
	 * @param img		Image (3-channel color image)
	 * @param ver		Vertical graph (Nx1 float-element matrix)
	 * @param hor		Horizontal graph (1xN float-element matrix)
	 * @param filename	output file name
	 */
	void outputImageWithHorizontalAndVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const cv::Mat& hor, const string& filename, int flag, int lineWidth) {
		int graphSize = std::max(100.0, std::max(img.rows, img.cols) * 0.3);

		cv::Mat result(img.rows + graphSize + 3, img.cols + graphSize + 3, CV_8UC3, cv::Scalar(255, 255, 255));

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
			int x1 = img.cols + (ver.at<float>(r, 0) - min_ver) / (max_ver - min_ver) * graphSize;
			int x2 = img.cols + (ver.at<float>(r + 1, 0) - min_ver) / (max_ver - min_ver) * graphSize;

			cv::line(result, cv::Point(x1, r), cv::Point(x2, r + 1), cv::Scalar(0, 0, 0), 1, cv::LINE_8);

			if (flag & GRAPH_LOCAL_MINIMUM && r > 0) {
				if (ver.at<float>(r, 0) < ver.at<float>(r - 1, 0) && ver.at<float>(r, 0) < ver.at<float>(r + 1, 0)) {
					cv::line(result, cv::Point(0, r), cv::Point(img.cols - 1, r), cv::Scalar(0, 0, 255), lineWidth);
				}
			}
		}

		// draw horizontal graph
		for (int c = 0; c < img.cols - 1; ++c) {
			//int y1 = result.rows - 1 - (hor.at<float>(0, c) - min_hor) / (max_hor - min_hor) * graphSize;
			//int y2 = result.rows - 1 - (hor.at<float>(0, c + 1) - min_hor) / (max_hor - min_hor) * graphSize;
			int y1 = img.rows + (hor.at<float>(0, c) - min_hor) / (max_hor - min_hor) * graphSize;
			int y2 = img.rows + (hor.at<float>(0, c + 1) - min_hor) / (max_hor - min_hor) * graphSize;

			cv::line(result, cv::Point(c, y1), cv::Point(c + 1, y2), cv::Scalar(0, 0, 0), 1, cv::LINE_8);

			if (flag & GRAPH_LOCAL_MINIMUM && c > 0) {
				if (hor.at<float>(0, c) < hor.at<float>(0, c - 1) && hor.at<float>(0, c) < hor.at<float>(0, c + 1)) {
					cv::line(result, cv::Point(c, 0), cv::Point(c, img.rows - 1), cv::Scalar(0, 0, 255), lineWidth);
				}
			}
		}

		cv::imwrite(filename, result);
	}

}