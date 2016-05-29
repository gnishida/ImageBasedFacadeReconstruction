#pragma once

#include <opencv2/opencv.hpp>

using namespace std;

namespace cvutils {
	
	static int LOCAL_MINIMUM = 1;
	static int LOCAL_MAXIMUM = 2;

	float max(const cv::Mat& mat);
	float min(const cv::Mat& mat);
	void write(const string& filename, const cv::Mat& mat);
	cv::Mat read(const string& filename);

	double msd(const cv::Mat& img1, const cv::Mat& img2);
	bool isLocalMinimum(const cv::Mat& mat, int index, int num);
	bool isLocalMaximum(const cv::Mat& mat, int index, int num);
	vector<int> getPeak(const cv::Mat& mat, bool smooth, int sigma, int flag = 1, int width = 1);
	float getMostPopularValue(const cv::Mat& h_max, float sigma, float min_value);
	void outputImageWithVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const string& filename);
	void outputImageWithHorizontalGraph(const cv::Mat& img, const cv::Mat& hor, const string& filename);
	void outputImageWithHorizontalAndVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const cv::Mat& hor, const string& filename, int flag = 0, int lineWidth = 1);
}

