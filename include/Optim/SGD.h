#pragma once
#include "Loss/Loss.h"

namespace engine {
	DLL_EXPORT void sgd_update(std::vector<Node*> update_nodes, float learning_rate);

	class DLL_EXPORT SGD {
	public:
		void update(std::vector<Node*> update_nodes, float learning_rate);

		void minimize(Loss* loss);
	private:

	};
}