#pragma once 
#include "Node.h"

namespace engine {
	class DLL_EXPORT RNN : public Node {
	public:
		RNN(Node* input,int t=1);

		void forward();
		void backward();
	private:
		Eigen::MatrixXf _hidden;
		int _maxStep = 0;
	};
}