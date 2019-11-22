#pragma once
#include "Node.h"

namespace engine {
	class DLL_EXPORT Sigmoid : public Node {
	public:
		Sigmoid(Node* node);

		virtual void forward(/*const Eigen::MatrixXf& value*/);
		virtual void backward();

	private:
		Eigen::MatrixXf _impl(const Eigen::MatrixXf& x);

	private:
		Node* _node = nullptr;
		//sigmoid的偏导
		Eigen::MatrixXf _partial;
	};
}