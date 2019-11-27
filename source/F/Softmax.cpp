#include "F/Softmax.h"
#include <numeric>
#include <execution>

namespace engine {

	Softmax::Softmax(Node* node)
		:Node(std::vector<Node*>({ node }))
		, _node(node)
	{
		_name = "softmax";
	}

	void Softmax::forward(/*const Eigen::MatrixXf& value*/)
	{
		_value = _impl(_node->getValue());
	}

	void Softmax::backward()
	{

	}

	Eigen::MatrixXf Softmax::_impl(const Eigen::MatrixXf& x)
	{
		if (x.cols() > 1) {
			auto m = x.rowwise().maxCoeff();
			auto v = x - m;
			auto e = v.array().exp();
			auto s = e.sum();
			return v / s;
		}
		else {
			float m = x.maxCoeff();
			auto v = x - Eigen::MatrixXf(x.rows(), x.cols()).setConstant(m);
			auto e = v.array().exp();
			auto s = e.sum();
			return v / s;
		}
	}

}