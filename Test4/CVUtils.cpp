#include "CVUtils.h"

namespace cvutils {

	float max(const cv::Mat& mat) {
		cv::Mat result;

		cv::reduce(mat, result, 0, CV_REDUCE_MAX);
		cv::reduce(result, result, 1, CV_REDUCE_MAX);

		return result.at<float>(0, 0);
	}

}