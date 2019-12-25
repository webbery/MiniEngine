#include "Loss/MSE.h"
#include "Debug.h"

namespace engine {

	MSE::MSE(Node* y, Node* y_hat)
		:Loss(y,y_hat)
	{
		_name = "MSE";
	}

	void MSE::forward(/*const Eigen::MatrixXf& value*/)
	{
		PRINT_SIZE(_y->getValue());
		PRINT_SIZE(_y_hat->getValue());
		_diff = _y->getValue() - _y_hat->getValue();
		//std::cout << "MSE: " << _diff.rows() << ", " << _diff.cols() << std::endl;
		auto diff2 = _diff.cwiseProduct(_diff);
		auto v = Eigen::MatrixXf(1, 1);
		v << (diff2).mean();
		_value = v;
	}

	void MSE::backward()
	{
		auto r = _y_hat->getValue().rows();
		//std::cout << "R: "<<r<<"----\n"<<_diff << std::endl;
		//反向传播的起点
		_gradients[_y] = _diff * (2.f / r);
		//std::cout << "======y========\n"<< _gradients[_y];
		_gradients[_y_hat] = _diff * (-2.f / r);
		//std::cout << "\n=======y_hat=======\n"<< _gradients[_y_hat];
	}
}