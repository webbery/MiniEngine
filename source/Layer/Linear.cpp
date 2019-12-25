#include "Layer/Linear.h"
#include "Debug.h"

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
		PRINT_SIZE(_nodes->getValue());
		PRINT_SIZE(_bias->getValue());
		PRINT_SIZE(_weights->getValue());
		auto b = _bias->getValue();
		if (b.cols() == 1) {
			//广播
			_value = (_nodes->getValue() * _weights->getValue()).rowwise() + Eigen::VectorXf(_bias->getValue()).transpose();
		}
		else {
			Eigen::MatrixXf t(_nodes->getValue() * _weights->getValue());
			size_t totalRows = t.rows() * b.cols();
			Eigen::MatrixXf lmat(totalRows, t.cols());
			PRINT_SIZE(lmat);
			for (int i = 0; i < b.cols(); ++i) {
				lmat.block(i * t.rows(), 0, t.rows(), t.cols()) = t;
			}
			auto r=b.transpose();
			Eigen::MatrixXf rmat(totalRows, t.cols());
			PRINT_SIZE(r);
			for (int i = 0; i < b.cols(); ++i) {
				Eigen::MatrixXf m(t.rows(), t.cols());
				PRINT_SIZE(m);
				for (size_t j = 0; j < t.rows(); ++j) {
					m.row(j) = r.row(i);
				}
				rmat.block(i*t.rows(), 0, t.rows(), t.cols()) = m;
			}
			_value = lmat + rmat;
		}
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

	void Linear::update(std::function<Node* (Node*)> executor)
	{
		_weights = executor(_weights);
		_bias = executor(_bias);
	}

}