#include "Utils.h"
#include <math.h>
#include <algorithm>

namespace utils {
	const float M_PI = 3.1415926535;

	float gause(float u, float sigma) {
		return 1.0f / 2.0f / M_PI / sigma / sigma * expf(-u * u / 2.0f / sigma / sigma);
	}

	/**
	 * ２つの数列がベストマッチするよう、２つ目の数列を変換する。
	 */
	std::vector<int> findBestAssignment(const std::vector<int>& labels1, const std::vector<int>& labels2) {
		int cardinality = 0;
		for (int i = 0; i < labels1.size(); ++i) {
			if (labels1[i] > cardinality) cardinality = labels1[i];
		}
		for (int i = 0; i < labels2.size(); ++i) {
			if (labels2[i] > cardinality) cardinality = labels2[i];
		}
		cardinality++;

		std::vector<int> mapping(cardinality);
		for (int i = 0; i < mapping.size(); ++i) {
			mapping[i] = i;
		}

		int min_dist = std::numeric_limits<int>::max();
		std::vector<int> min_mapping;
		do {
			int dist = 0;
			for (int i = 0; i < labels1.size(); ++i) {
				if (mapping[labels1[i]] != labels2[i]) {
					dist++;
				}
			}

			if (dist < min_dist) {
				min_dist = dist;
				min_mapping = mapping;
			}

		} while (std::next_permutation(mapping.begin(), mapping.end()));

		return min_mapping;
	}
}