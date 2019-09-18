#pragma once
#include <string>
#include <vector>
#include <map>
#ifdef __USE_OPENCL__
#include <boost/compute.hpp>
#include <boost/compute/interop/eigen.hpp>
#else
#include <Eigen/Core>
#endif

namespace engine {
	class Node {
	public:
		Node() {}

		Node(const std::vector<Node*>& inputs) {
			for (auto node : inputs) {
				node->addNode2Output(this);
			}
		}
		virtual ~Node() {}

		virtual void forward(const Eigen::MatrixXf& value) = 0;

		virtual void backward() = 0;

		Eigen::MatrixXf getValue() { return _value; }
		void setValue(const Eigen::MatrixXf& value) { _value = value; }

	public:
		void addNode2Output(Node* pNode) { _outputs.emplace_back(pNode); }
		Eigen::MatrixXf getGradient(Node* name) { return _gradients[name]; }

	protected:
		Eigen::MatrixXf _value;
		std::vector<Node*> _inputs;
		std::vector<Node*> _outputs;
		std::map<Node*, Eigen::MatrixXf> _gradients;
	};

	class Input : public Node {//此类等价于placeholder
	public:
		Input(const char* name/*, const std::vector<Node*>& inputs*/);

		virtual void forward(const Eigen::MatrixXf& value);

		virtual void backward();

	private:
		std::string _name;
		Eigen::MatrixXf _value;
	};

	class Linear : public Node {
	public:
		Linear(Node* nodes, Node* weights, Node* bias);

		virtual void forward(const Eigen::MatrixXf& value);

		virtual void backward();

	private:
		Node* _nodes = nullptr;
		Node* _weights = nullptr;
		Node* _bias = nullptr;
	};

	class Sigmoid : public Node {
	public:
		Sigmoid(Node* node);

		virtual void forward(const Eigen::MatrixXf& value);
		virtual void backward();

	private:
		Eigen::MatrixXf _impl(const Eigen::MatrixXf& x);

	private:
		Node* _node = nullptr;
		//sigmoid的偏导
		Eigen::MatrixXf _partial;
	};

	class MSE : public Node {
	public:
		MSE(Node* y, Node* y_hat);

		virtual void forward(const Eigen::MatrixXf& value);
		virtual void backward();

	private:
		Node* _y;
		Node* _y_hat;
		Eigen::MatrixXf _diff;
	};
}