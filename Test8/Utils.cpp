#include "Utils.h"
#include <math.h>

namespace utils {
	const float M_PI = 3.1415926535;

	float gause(float u, float sigma) {
		return 1.0f / 2.0f / M_PI / sigma / sigma * expf(-u * u / 2.0f / sigma / sigma);
	}

}