#pragma once
#include "Config.h"
#include <string>
#include <vector>
#include <map>
#include <iostream>

namespace engine {
	class DLL_EXPORT Node {
	public:
		Node() {}

		Node(const std::vector<Node*>& inputs)
		{
			init(inputs, inputs);
		}
		virtual ~Node() {}

		virtual void forward() = 0;

		//反向计算梯度
		virtual void backward() = 0;

		//根据梯度进行参数更新
		virtual void update(std::function<Node* (Node*)> executor) {}

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