#include "Layer/Dropout.h"

namespace engine {

	Dropout::Dropout(Node* node, float seed/*=0.5*/)
		:Node(std::vector<Node*>({ node }))
		, _distribution(seed)
		,_node(node)
	{
		auto m = node->getValue();
		_r.resize(m.cols(), m.rows());
		reset();
	}

	void Dropout::forward(/*const Eigen::MatrixXf& value*/)
	{
		_value = _node->getValue() * _r;
	}

	void Dropout::backward()
	{
		for (auto node : _outputs) {
			auto grad = node->getGradient(this);
			_gradients[_node] = Eigen::Matrix2f::Zero(grad.rows(),grad.cols());
		}
		reset();
	}

	void Dropout::reset()
	{
		for (size_t indx = 0; indx < _r.cols(); ++indx) {
			for (size_t row = 0; row < _r.rows(); ++row) {
				_r(row, indx) = (_distribution(_generator) ? 1 : 0);
			}
		}
	}

}