#include "Layer/Dropout.h"

namespace engine {

	Dropout::Dropout(Node* node, float seed/*=0.5*/)
		:Node(std::vector<Node*>({ node }))
		, _distribution(seed)
		,_node(node)
	{
		_rescale = 1 / (1.0f - seed);
		_name = "dropout";
	}

	void Dropout::forward(/*const Eigen::MatrixXf& value*/)
	{
		auto m = _node->getValue();
		//std::cout << m.cols() << ", " << m.rows() << std::endl;
		_r = Eigen::MatrixXf(m.cols(), m.cols());
		reset();

		_value = _node->getValue()*_r;
		//std::cout << _value.cols() << ", " << _value.rows() << std::endl;
	}

	void Dropout::backward()
	{
		for (auto node : _outputs) {
			auto grad = node->getGradient(this);
			_gradients[_node] = grad * _r;// Eigen::Matrix2f::Zero(grad.rows(), grad.cols());
		}
	}

	void Dropout::reset()
	{
		for (size_t indx = 0; indx < _r.cols(); ++indx) {
			for (size_t row = 0; row < _r.rows(); ++row) {
				_r(row, indx) = _rescale *(_distribution(_generator) ? 1 : 0);
			}
		}
	}

}