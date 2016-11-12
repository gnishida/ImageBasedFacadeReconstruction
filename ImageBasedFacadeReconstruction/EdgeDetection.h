#pragma once

#include <vector>
#include <tuple>
#include <opencv2/calib3d.hpp>
#include <glm/glm.hpp>

namespace ed {

	std::pair<float, float> detectHorizontalAndVerticalDominantOrientation(const cv::Mat& img, float angle_threshold, bool remove_diagonal_edges, bool use_magnitude_as_weight, bool smooth_accum, float voteThresholdRatio);
	std::vector<std::tuple<glm::vec2, glm::vec2, int, float>> detectHorizontalAndVerticalEdges(const cv::Mat& img, float angle_threshold, int edge_detector, bool remove_diagonal_edges, bool use_magnitude_as_weight, bool smooth_accum, int topN);
	void warp(cv::Mat img, float hori, float vert, cv::Mat& warped);
	void removeDiagonalEdges(const cv::Mat& grayImg, cv::Mat& edgeImg, float angle_threshold);

	// helper functions
	void gradientOrientation(cv::Mat img, cv::Mat& gradOri);
	void gradientMagnitude(cv::Mat img, cv::Mat& grad);
	void autoCanny(const cv::Mat& grayImg, cv::Mat& edgeImg, float sigma = 0.33f);
	void autoCanny2(const cv::Mat& grayImg, cv::Mat& edgeImg);
	double median(const cv::Mat& grayImg);
	float getVerticalAccumMax(const cv::Mat& accum, float max_degree);
	float getHorizontalAccumMax(const cv::Mat& accum, float max_degree);
	
}

