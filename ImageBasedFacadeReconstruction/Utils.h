#pragma once

#include <vector>
#include <map>
#include <algorithm>

namespace utils {

	float gause(float u, float sigma);	
	float median(std::vector<float> list);
	float stddev(std::vector<float> list);
	float mean(std::vector<float> list);

	std::vector<int> findBestAssignment(const std::vector<int>& labels1, const std::vector<int>& labels2);
}