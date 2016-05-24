#pragma once

#include <opencv2/opencv.hpp>

using namespace std;

namespace cvutils {
	
	static int GRAPH_LOCAL_MINIMUM = 1;

	float max(const cv::Mat& mat);
	float min(const cv::Mat& mat);
	void write(const string& filename, const cv::Mat& mat);
	cv::Mat read(const string& filename);

	bool localMinimum(const cv::Mat& mat, int index, int num);
	void outputImageWithHorizontalAndVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const cv::Mat& hor, const string& filename, int flag = 0);
}

