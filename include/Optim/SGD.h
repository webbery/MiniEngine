#pragma once
#include "Node.h"

namespace engine {
	DLL_EXPORT void sgd_update(std::vector<Node*> update_nodes, float learning_rate);

	class DLL_EXPORT SGD {
	public:
		void update(std::vector<Node*> update_nodes, float learning_rate);

		void minimize(Node* loss);
	private:

	};
}