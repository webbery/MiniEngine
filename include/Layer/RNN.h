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
		Eigen::MatrixXf _bias;
		Eigen::MatrixXf _weight;
		Eigen::MatrixXf _U;
		Eigen::MatrixXf _V;
		Eigen::MatrixXf _c;
		Node* _node;
		int _maxStep = 0;
	};
}