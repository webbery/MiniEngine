#pragma once
#include "Node.h"

namespace engine {
	class DLL_EXPORT ReLu : public Node {
	public:
		ReLu(Node* node);

		virtual void forward(/*const Eigen::MatrixXf& value*/);
		virtual void backward();
	};
}