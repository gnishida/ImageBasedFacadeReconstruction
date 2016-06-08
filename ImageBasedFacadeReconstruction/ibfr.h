#pragma once

#include <opencv2/opencv.hpp>

class Tile {
public:
	//int width;
	//int height;
	cv::Mat image;
	int cluster_id;

public:
	Tile() : cluster_id(-1) {}
	Tile(const cv::Mat& image) : image(image) {}
};