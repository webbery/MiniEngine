#pragma once
#include "Node.h"

namespace engine {
	class DLL_EXPORT Softmax : public Node {
	public:
		Softmax(Node* node);

		virtual void forward(/*const Eigen::MatrixXf& value*/);
		virtual void backward();

	private:
		Eigen::MatrixXf _impl(const Eigen::MatrixXf& x);
	private:
		Node* _node = nullptr;
	};
}