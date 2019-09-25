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

#ifdef MiniEngine_EXPORTS
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __declspec(dllimport)
#endif

namespace engine {
	class DLL_EXPORT Node {
	public:
		Node() {}

		Node(const std::vector<Node*>& inputs)
		:_inputs(inputs)
		{
			for (auto node : inputs) {
				node->addNode2Output(this);
			}
		}
		virtual ~Node() {}

		virtual void forward(/*const Eigen::MatrixXf& value*/) = 0;

		virtual void backward() = 0;

		Eigen::MatrixXf getValue() { return _value; }
		void setValue(const Eigen::MatrixXf& value) { _value = value; }

	public:
		void addNode2Output(Node* pNode) { _outputs.emplace_back(pNode); }
		Eigen::MatrixXf getGradient(Node* name) { return _gradients[name]; }
		std::vector<Node*>& getOutputs() { return _outputs; }
		std::vector<Node*> getInputs() { return _inputs; }
		std::string name() { return _name; }

	protected:
		Eigen::MatrixXf _value;
		std::vector<Node*> _inputs;
		std::vector<Node*> _outputs;
		std::map<Node*, Eigen::MatrixXf> _gradients;
		std::string _name;
	};

	class DLL_EXPORT Input : public Node {//此类等价于placeholder
	public:
		Input(const char* name,size_t rows=0,size_t cols=0);

		virtual void forward(/*const Eigen::MatrixXf& value*/);

		virtual void backward();
	};

	class DLL_EXPORT Linear : public Node {
	public:
		Linear(Node* nodes, Node* weights, Node* bias);

		virtual void forward(/*const Eigen::MatrixXf& value*/);

		virtual void backward();

	private:
		Node* _nodes = nullptr;
		Node* _weights = nullptr;
		Node* _bias = nullptr;
	};

	class DLL_EXPORT Sigmoid : public Node {
	public:
		Sigmoid(Node* node);

		virtual void forward(/*const Eigen::MatrixXf& value*/);
		virtual void backward();

	private:
		Eigen::MatrixXf _impl(const Eigen::MatrixXf& x);

	private:
		Node* _node = nullptr;
		//sigmoid的偏导
		Eigen::MatrixXf _partial;
	};

	class DLL_EXPORT MSE : public Node {
	public:
		MSE(Node* y, Node* y_hat);

		virtual void forward(/*const Eigen::MatrixXf& value*/);
		virtual void backward();

	private:
		Node* _y;
		Node* _y_hat;
		Eigen::MatrixXf _diff;
	};

	DLL_EXPORT std::vector<Node*> topological_sort(Node* input_nodes);

	DLL_EXPORT void train_one_batch(std::vector<Node*>& graph);

	DLL_EXPORT void sgd_update(std::vector<Node*> update_nodes, float learning_rate = 1e-2);

}// namespace engine