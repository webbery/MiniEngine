#pragma once
#ifdef __USE_OPENCL__
#include <boost/compute.hpp>
#include <boost/compute/interop/eigen.hpp>
#else
#include <Eigen/Core>
#endif
#include <string>
#include <vector>
#include <map>

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
		{
			init(inputs, inputs);
		}
		virtual ~Node() {}

		virtual void forward(/*const Eigen::MatrixXf& value*/) = 0;

		virtual void backward() = 0;

		Eigen::MatrixXf getValue() { return _value; }
		void setValue(const Eigen::MatrixXf& value) { _value = value; }

		void init(const std::vector<Node*>& inputs, const std::vector<Node*>& outputs) {
			_inputs = inputs;
			for (auto node : outputs) {
				node->addNode2Output(this);
			}
		}

	public:
		void addNode2Output(Node* pNode) { _outputs.emplace_back(pNode); }
		Eigen::MatrixXf getGradient(Node* name) {
			return _gradients[name];
		}
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

}