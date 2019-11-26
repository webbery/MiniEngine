#pragma once
#include "Node.h"

namespace engine {

	class DLL_EXPORT Loss : public Node {
	public:
		Loss(Node* y, Node* y_hat) 
			:Node(std::vector<Node*>({ y,y_hat }))
			, _y(y), _y_hat(y_hat)
		{
			reset();
		}

		Eigen::MatrixXf operator + (const Eigen::MatrixXf& m) {
			return _loss + m;
		}

		void operator += (const Eigen::MatrixXf& m) {
			_loss += m;
		}

		operator float() {
			return _loss(0, 0);
		}

		void reset() {
			_loss.setZero();
		}
	protected:
		Node* _y;
		Node* _y_hat;

		Eigen::MatrixXf _loss = Eigen::MatrixXf(1, 1);
	};
}