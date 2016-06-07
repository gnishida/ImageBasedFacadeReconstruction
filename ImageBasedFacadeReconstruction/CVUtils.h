#pragma once

#include <opencv2/opencv.hpp>

using namespace std;

namespace cvutils {
	
	static int LOCAL_MINIMUM = 1;
	static int LOCAL_MAXIMUM = 2;

	double max(const cv::Mat& mat);
	double min(const cv::Mat& mat);
	vector<double> mean(const cv::Mat& mat);
	void write(const string& filename, const cv::Mat& mat);
	cv::Mat read(const string& filename);

	double msd(const cv::Mat& img1, const cv::Mat& img2);
	double corr(const cv::Mat& img1, const cv::Mat& img2);
	double get(const cv::Mat& mat, int r, int c);
	bool isLocalMinimum(const cv::Mat& mat, int index, int num);
	bool isLocalMaximum(const cv::Mat& mat, int index, int num);
	vector<int> getPeak(const cv::Mat& mat, bool smooth, int sigma, int flag = 1, int width = 1);
	float getMostPopularValue(const cv::Mat& h_max, float sigma, float min_value);
	void outputImageWithVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const string& filename, int flag = 0, int continuous_num = 1, int lineWidth = 1);
	void outputImageWithHorizontalGraph(const cv::Mat& img, const cv::Mat& hor, const string& filename, int flag = 0, int continuous_num = 1, int lineWidth = 1);
	void outputImageWithHorizontalAndVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const cv::Mat& hor, const string& filename, int flag = 0, int continuous_num = 1, int lineWidth = 1);
}

