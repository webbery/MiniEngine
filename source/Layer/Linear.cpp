#include "Layer/Linear.h"

namespace engine {
	Linear::Linear(Node* nodes, Node* weights, Node* bias)
		:Node(std::vector<Node*>({ nodes,weights,bias }))
		, _nodes(nodes)
		, _weights(weights)
		, _bias(bias)
	{
		_name = "Linear";
		//_gradients[this] = Eigen::MatrixXf(rows, cols).setRandom();
	}

	void Linear::forward(/*const Eigen::MatrixXf&*/)
	{
		//std::cout <<"Linear: "<< _nodes->getValue().rows() <<", "<< _nodes->getValue().cols() 
		//	<<"\tWeight: "<< _weights->getValue().rows()<<", "<< _weights->getValue().cols()
		//	<<"\tBias: "<< _bias->getValue().rows()<<", "<< _bias->getValue().cols()<< std::endl;
		//广播
		//auto WX = _nodes->getValue() * _weights->getValue();
		//Eigen::VectorXf vec(_bias->getValue());
		_value = (_nodes->getValue() * _weights->getValue()).rowwise() + Eigen::VectorXf(_bias->getValue()).transpose();
	}

	void Linear::backward()
	{
		for (auto node : _outputs) {
			auto grad = node->getGradient(this);
			//std::cout << "Linear backward: " << _nodes->getValue().rows() << ", " << _nodes->getValue().cols()
			//	<< "\tgrad: " << grad.rows() << ", " << grad.cols() << std::endl;
			_gradients[_weights] = _nodes->getValue().transpose() * grad;
			//按列求和,变为一行
			//_gradients[_bias] = grad;// .rowwise().sum();
			//std::cout << "Linear backward: " << _nodes->getValue().rows() << ", " << _nodes->getValue().cols()
			//	<< "\tgrad: " << _gradients[_bias].rows() << ", " << _gradients[_bias].cols() << std::endl;
			_gradients[_bias] = grad.colwise().sum().transpose();
			//std::cout << "grad: " << grad.rows() << ", " << grad.cols() << "\tbias: " << _gradients[_bias].rows() << ", " << _gradients[_bias].cols() << std::endl;
			_gradients[_nodes] = grad * _weights->getValue().transpose();
		}
	}
}