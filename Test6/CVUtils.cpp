#include "CVUtils.h"
#include <fstream>
#include <iostream>
#include <regex>

using namespace std;

namespace cvutils {

	float max(const cv::Mat& mat) {
		cv::Mat result;

		cv::reduce(mat, result, 0, CV_REDUCE_MAX);
		cv::reduce(result, result, 1, CV_REDUCE_MAX);

		return result.at<float>(0, 0);
	}

	float min(const cv::Mat& mat) {
		cv::Mat result;

		cv::reduce(mat, result, 0, CV_REDUCE_MIN);
		cv::reduce(result, result, 1, CV_REDUCE_MIN);

		return result.at<float>(0, 0);
	}

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

}