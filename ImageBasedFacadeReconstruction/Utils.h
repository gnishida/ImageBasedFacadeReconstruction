#pragma once

#include <vector>
#include <map>

namespace utils {

	float gause(float u, float sigma);

	std::vector<int> findBestAssignment(const std::vector<int>& labels1, const std::vector<int>& labels2);
}