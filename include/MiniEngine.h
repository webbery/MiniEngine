#pragma once
#include "Node.h"

namespace engine {
	
	class DLL_EXPORT Input : public Node {//此类等价于placeholder
	public:
		Input(const char* name,size_t rows=0,size_t cols=0);

		virtual void forward(/*const Eigen::MatrixXf& value*/);

		virtual void backward();
	};

	DLL_EXPORT std::vector<Node*> topological_sort(Node* input_nodes);

	DLL_EXPORT void train_one_batch(std::vector<Node*>& graph);

	DLL_EXPORT void sgd_update(std::vector<Node*> update_nodes, float learning_rate = 1e-2);

}// namespace engine