#pragma once
#include "Config.h"
#include <vector>

namespace engine {
	DLL_EXPORT std::vector<Eigen::MatrixXf> make_blob(size_t samples, size_t features = 2, size_t centers = 3,float* pbox =nullptr, float* cstd = nullptr);
}