#pragma once
#include "Loss.h"

namespace engine {
	class DLL_EXPORT MSE final: public Loss{
	public:
		MSE(Node* y, Node* y_hat);

		virtual void forward(/*const Eigen::MatrixXf& value*/);
		virtual void backward();

	private:
		Eigen::MatrixXf _diff;
	};
}