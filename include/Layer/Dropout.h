#pragma once
#include "Node.h"
#include <random>

namespace engine {
	class DLL_EXPORT Dropout : public Node {
	public:
		Dropout(Node* nodes,float seed=0.5);

		virtual void forward(/*const Eigen::MatrixXf& value*/);

		virtual void backward();

	private:
		void reset();
	private:
		std::default_random_engine _generator;
		std::bernoulli_distribution _distribution;
		Eigen::MatrixXf _r;
		Node* _node = nullptr;
		float _rescale = 0.5;
	};
}