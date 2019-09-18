#include "MiniEngine.h"

namespace engine {

	Input::Input(const char* name/*, const std::vector<Node* >& inputs*/)
		:_name(name)
	{
		
	}

	void Input::forward(const Eigen::MatrixXf& value) {
		_value = value;
	}

	void Input::backward() {
		for (auto node : _outputs) {
			_gradients[this] = node->getGradient(this);
		}
	}

	Linear::Linear(Node* nodes, Node* weights, Node* bias)
		:Node(std::vector<Node*>({nodes,weights,bias}))
		,_nodes(nodes)
		,_weights(weights)
		,_bias(bias)
	{
	}

	void Linear::forward(const Eigen::MatrixXf& )
	{
		_value = _nodes->getValue() * _weights->getValue() + _bias->getValue();
	}

	void Linear::backward()
	{
		for (auto node : _outputs) {
			auto grad = node->getGradient(this);
			_gradients[_weights] = _nodes->getValue().transpose() * grad;
			//按列求和,变为一行
			_gradients[_bias] = grad.colwise().sum();
			_gradients[_nodes] = grad * _weights->getValue().transpose();
		}
	}

	Sigmoid::Sigmoid(Node* node)
		:Node(std::vector<Node*>({node}))
		,_node(node)
	{
	}

	void Sigmoid::forward(const Eigen::MatrixXf& )
	{
		_value = _impl(_node->getValue());
	}

	void Sigmoid::backward()
	{
		auto y = _value;
		_partial = y-y*y;

		for (auto node : _outputs) {
			auto grad = node->getGradient(this);
			_gradients[_node] = grad * _partial;
		}
	}

	Eigen::MatrixXf Sigmoid::_impl(const Eigen::MatrixXf& x)
	{
		return (x.array().exp() + 1).inverse();
	}

	MSE::MSE(Node* y, Node* y_hat)
		:Node(std::vector<Node*>({y,y_hat}))
		,_y(y),_y_hat(y_hat)
	{
	}

	void MSE::forward(const Eigen::MatrixXf& value)
	{
		_diff = _y->getValue() - _y_hat->getValue();
		_value = Eigen::MatrixXf((_diff*_diff).mean());
	}

	void MSE::backward()
	{
		auto r = _y_hat->getValue().rows();

	}

}