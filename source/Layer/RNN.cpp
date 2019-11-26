#include "Layer/RNN.h"

namespace engine {

	RNN::RNN(Node* input, int t)
		:Node({input})
		, _maxStep(t)
	{
		_name = "rnn";
	}

	void RNN::forward()
	{
		//前向计算输入的值,然后计算hidden和输出
		for (size_t step = 0; step < _maxStep; ++step) {

		}
	}

	void RNN::backward()
	{

	}

}