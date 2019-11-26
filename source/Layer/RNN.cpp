#include "Layer/RNN.h"

namespace engine {

	RNN::RNN(Node* input, int t)
		:Node({input})
		, _node(input)
		, _maxStep(t)
	{
		_name = "rnn";
		auto& x = input->getValue();
		//hidden比输入多一个初始值
		_hidden.setZero(x.rows() + 1, x.cols());
		_bias.setZero(x.rows(), x.cols());
		_weight.setZero(x.rows(), x.cols());
	}

	void RNN::forward()
	{
		auto& x = _node->getValue();
		//前向计算输入的值,然后计算hidden和输出
		for (size_t t = 0; t < x.rows(); ++t) {
			auto a_t = _bias + _weight * _hidden.row(t) + _U * x.row(t);
			//_hidden.row(t + 1) = Eigen::tanh(a_t);
		}
	}

	void RNN::backward()
	{

	}

}