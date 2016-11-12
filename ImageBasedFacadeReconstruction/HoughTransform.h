#pragma once

#include <opencv/cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>

class HoughTransform {
protected:
	HoughTransform() {}

public:
	static void computeAccumulator(cv::Mat image, cv::Mat& accum, bool use_magnitude_as_weight, cv::Mat grad);
	static void suppressNonMaximum(cv::Mat accum, cv::Mat& result, int size, float threshold = 0);
	static void saveAccum(const std::string& filename, const cv::Mat& accum);
	static void computeLines(const cv::Mat& accum, std::vector<cv::Vec4b>& lines);
	static void visualizeAccum(const cv::Mat& image, const cv::Mat& accum, int max_degree, const std::string& filename);
	static bool isLocalMaximum(const cv::Mat& mat, int r, int c, int size);
};

#if 0
namespace ht {

	float warpImageByDominantOrientation(cv::Mat& image, float true_hori, float true_vert);
	void houghTransform(const cv::Mat& image, const cv::Size& kernel, cv::Mat& accum);
	void getDominantOrientation(const cv::Mat& image, const cv::Size& kernel, int max_degree, float threshold_ratio, double& hori, double& vert);
	float getVerticalOrientation(const cv::Mat& accum, int max_degree, float threshold_ratio);
	float getHorizontalOrientation(const cv::Mat& accum, int max_degree, float threshold_ratio);
	float getVerticalAccumMax(const cv::Mat& accum, float max_degree);
	float getHorizontalAccumMax(const cv::Mat& accum, float max_degree);
	bool isLocalMaximum(const cv::Mat& mat, int r, int c, int size);

	void autoCanny(const cv::Mat& grayImg, cv::Mat& edgeImg, float sigma = 0.33f);
	void autoCanny2(const cv::Mat& grayImg, cv::Mat& edgeImg, float sigma = 0.33f);
	double median(const cv::Mat& grayImg);

	void saveImage(const cv::Mat& image, const std::string& filename);
	void saveHistogram(const cv::Mat& mat, const std::string& filename);

	void visualizeAccum(const cv::Mat& image, const cv::Mat& accum, int max_degree, float threshold_ratio, const std::string& filename);
}
#endif