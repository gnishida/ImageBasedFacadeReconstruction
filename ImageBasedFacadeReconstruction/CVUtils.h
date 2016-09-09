#pragma once

#include <opencv2/opencv.hpp>

namespace cvutils {
	
	static int LOCAL_MINIMUM = 1;
	static int LOCAL_MAXIMUM = 2;

	double max(const cv::Mat& mat);
	double min(const cv::Mat& mat);
	std::vector<double> mean(const cv::Mat& mat);
	void write(const std::string& filename, const cv::Mat& mat);
	cv::Mat read(const std::string& filename);

	double msd(const cv::Mat& img1, const cv::Mat& img2);
	double corr(const cv::Mat& img1, const cv::Mat& img2);
	double get(const cv::Mat& mat, int r, int c);
	void grayScale(const cv::Mat& img, cv::Mat& grayImg);
	void blend(cv::Mat foreground, cv::Mat background, cv::Mat& blended);
	void grad(cv::Mat img, cv::Mat& grad);
	void median(cv::Mat img, int x, int y, cv::Mat& med);
	bool isLocalMinimum(const cv::Mat& mat, int index, int num);
	bool isLocalMaximum(const cv::Mat& mat, int index, int num);
	std::vector<int> getPeak(const cv::Mat& mat, bool smooth, int sigma, int flag = 1, int width = 1);
	float getMostPopularValue(const cv::Mat& h_max, float sigma, float min_value);
	float findNextMax(cv::Mat mat, int index, int& max_index);
	bool findNextMax(cv::Mat mat, int index, int dir, int& max_index, float& max_value);
	void outputImageWithVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const std::string& filename, int flag = 0, int continuous_num = 1, int lineWidth = 1);
	void outputImageWithHorizontalGraph(const cv::Mat& img, const cv::Mat& hor, const std::string& filename, int flag = 0, int continuous_num = 1, int lineWidth = 1);
	void outputImageWithHorizontalAndVerticalGraph(const cv::Mat& img, const cv::Mat& ver, const cv::Mat& hor, const std::string& filename, int flag = 0, int continuous_num = 1, int lineWidth = 1);

	float computeImageDist(const cv::Mat& img1, const cv::Mat& img2);
	void clusterImages(const std::vector<cv::Mat>& images, std::vector<int>& labels, std::vector<cv::Mat>& centers, int max_centers);
}

