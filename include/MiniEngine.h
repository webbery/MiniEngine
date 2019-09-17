#pragma once
#include <any>
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

		virtual void forward(const std::any& value) = 0;

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

		virtual void forward(const std::any& value);

		virtual void backward();

	private:
		Node* _nodes = nullptr;
		Node* _weights = nullptr;
		Node* _bias = nullptr;

	};
}