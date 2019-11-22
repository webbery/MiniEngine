#pragma once
#include "Node.h"

namespace engine {
	class DLL_EXPORT MSE : public Node {
	public:
		MSE(Node* y, Node* y_hat);

		virtual void forward(/*const Eigen::MatrixXf& value*/);
		virtual void backward();

	private:
		Node* _y;
		Node* _y_hat;
		Eigen::MatrixXf _diff;
	};
}