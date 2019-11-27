#pragma once
#include "Node.h"

namespace engine {
	class DLL_EXPORT Linear final: public Node {
	public:
		Linear(Node* nodes, Node* weights, Node* bias);

		virtual void forward(/*const Eigen::MatrixXf& value*/);

		virtual void backward();

		virtual void update(std::function<Node* (Node*)> executor);

	private:
		Node* _nodes = nullptr;
		Node* _weights = nullptr;
		Node* _bias = nullptr;
	};
}