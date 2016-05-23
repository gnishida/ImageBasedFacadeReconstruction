#pragma once

#include <opencv2/opencv.hpp>

using namespace std;

namespace cvutils {
	
	float max(const cv::Mat& mat);
	float min(const cv::Mat& mat);
	void write(const string& filename, const cv::Mat& mat);
	cv::Mat read(const string& filename);

}

