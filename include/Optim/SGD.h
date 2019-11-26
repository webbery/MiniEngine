#pragma once
#include "Node.h"

namespace engine {
	DLL_EXPORT void sgd_update(std::vector<Node*> update_nodes, float learning_rate);
}