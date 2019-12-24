#include "F/Softmax.h"
#include <numeric>
#include <execution>
#include <unsupported/Eigen/KroneckerProduct>
#include "Debug.h"

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
		auto y = _value;
		auto y2 = y.cwiseProduct(y);
		size_t size = std::min(y.rows(), y.cols());
		Eigen::MatrixXf I = Eigen::MatrixXf::Identity(y.rows(), y.cols());
		//y(delta(i,j)-y),	i==j时delta(i,j)为1,否则为0
		Eigen::MatrixXf partial = y.cwiseProduct(I) - y2;
		for (auto node : _outputs) {
			auto grad = node->getGradient(this);
			_gradients[_node] = grad.cwiseProduct(partial);
		}
	}

	Eigen::MatrixXf Softmax::_impl(const Eigen::MatrixXf& x)
	{
		if (x.cols() > 1) {
			auto m = x.rowwise().maxCoeff();
			Eigen::MatrixXf temp = Eigen::MatrixXf(m);
			temp.conservativeResize(x.rows(), x.cols());
			for (int c = 1; c < x.cols(); ++c) {
				temp.col(c) = m;
			}
			auto v = x - temp;
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