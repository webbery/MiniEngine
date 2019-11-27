#include "F/Sigmoid.h"

namespace engine {

	Sigmoid::Sigmoid(Node* node)
		:Node(std::vector<Node*>({ node }))
		, _node(node)
	{
		_name = "sigmoid";
	}

	void Sigmoid::forward(/*const Eigen::MatrixXf&*/)
	{
		_value = _impl(_node->getValue());
	}

	void Sigmoid::backward()
	{
		auto y = _value;
		//std::cout << "Sigmoid backward: y " << y.rows() << ", " << y.cols() << std::endl;
		auto y2 = y.cwiseProduct(y);
		//std::cout << y2.rows() << ", " << y2.cols() << std::endl;
		_partial = y - y2;

		for (auto node : _outputs) {
			auto grad = node->getGradient(this);
			//std::cout << "Sigmoid: " << grad.rows() << ", " << grad.cols()
			//	<< "\t partial: " << _partial.rows() << ", " << _partial.cols() << std::endl;
			_gradients[_node] = grad.cwiseProduct(_partial);
		}
	}

	Eigen::MatrixXf Sigmoid::_impl(const Eigen::MatrixXf& x)
	{
		return (-x.array().exp() + 1).inverse();
		//return 1.f / (1.f + exp(-x));
	}
}