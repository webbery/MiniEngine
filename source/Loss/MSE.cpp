#include "Loss/MSE.h"

namespace engine {

	MSE::MSE(Node* y, Node* y_hat)
		:Node(std::vector<Node*>({ y,y_hat }))
		, _y(y), _y_hat(y_hat)
	{
		_name = "MSE";
	}

	void MSE::forward(/*const Eigen::MatrixXf& value*/)
	{
		//std::cout << "MSE: y " << _y->getValue().rows() << ", " << _y->getValue().cols() 
		//	<< "\ty " << _y_hat->getValue().rows() << ", " << _y_hat->getValue().cols()<<  std::endl;
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