#pragma once

#include <opencv2/core.hpp>
#include <string>
#include "facade_segmentation.h"

namespace fe {

	void subdivideFacade(const cv::Mat& img, std::vector<float>& y_split, std::vector<float>& x_split, std::vector<std::vector<fs::WindowPos>>& win_rects);

}

