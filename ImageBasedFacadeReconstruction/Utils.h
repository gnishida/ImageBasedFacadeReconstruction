#pragma once

#include <vector>

namespace utils {

	float gause(float u, float sigma);

	void findBestAssignment(const std::vector<int>& labels1, std::vector<int>& labels2);
}