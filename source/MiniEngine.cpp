#include "MiniEngine.h"
#include <map>
#include <set>
#include <iostream>

namespace engine {
	Eigen::MatrixXf Node::getGradient(Node* name)
	{
		//std::cout << _name << std::endl;
		return _gradients[name];
	}
	Input::Input(const char* name,size_t rows,size_t cols)
	{
		_name = name;
		//随机初始化
		_value = Eigen::MatrixXf(rows, cols).setRandom();
		_gradients[this] = Eigen::MatrixXf(rows, cols).setRandom();
	}

	void Input::forward(/*const Eigen::MatrixXf& value*/) {
		//_value = value;
	}

	void Input::backward() {
		for (auto node : _outputs) {
			//std::cout << node->name() << std::endl;
			_gradients[this] = node->getGradient(this);
			//std::cout << _gradients[this] << std::endl;
			//std::cout << "Input " << node->name() << ": " << _gradients[this].rows() << ", " << _gradients[this].cols() << std::endl;
		}
	}

	Linear::Linear(Node* nodes, Node* weights, Node* bias)
		:Node(std::vector<Node*>({nodes,weights,bias}))
		,_nodes(nodes)
		,_weights(weights)
		,_bias(bias)
	{
		_name = "Linear";
		//_gradients[this] = Eigen::MatrixXf(rows, cols).setRandom();
	}

	void Linear::forward(/*const Eigen::MatrixXf&*/ )
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

	Sigmoid::Sigmoid(Node* node)
		:Node(std::vector<Node*>({node}))
		,_node(node)
	{
		_name = "Sigmoid";
	}

	void Sigmoid::forward(/*const Eigen::MatrixXf&*/ )
	{
		_value = _impl(_node->getValue());
	}

	void Sigmoid::backward()
	{
		auto y = _value;
		//std::cout << "Sigmoid backward: y " << y.rows() << ", " << y.cols() << std::endl;
		auto y2 = y.cwiseProduct(y);
		//std::cout << y2.rows() << ", " << y2.cols() << std::endl;
		_partial = y-y2;

		for (auto node : _outputs) {
			auto grad = node->getGradient(this);
			//std::cout << "Sigmoid: " << grad.rows() << ", " << grad.cols()
			//	<< "\t partial: " << _partial.rows() << ", " << _partial.cols() << std::endl;
			_gradients[_node] = grad.cwiseProduct(_partial);
		}
	}

	Eigen::MatrixXf Sigmoid::_impl(const Eigen::MatrixXf& x)
	{
		return (-x.array().exp() + 1).inverse();
		//return 1.f / (1.f + exp(-x));
	}

	MSE::MSE(Node* y, Node* y_hat)
		:Node(std::vector<Node*>({y,y_hat}))
		,_y(y),_y_hat(y_hat)
	{
		_name = "MSE";
	}

	void MSE::forward(/*const Eigen::MatrixXf& value*/)
	{
		//std::cout << "MSE: y " << _y->getValue().rows() << ", " << _y->getValue().cols() 
		//	<< "\ty " << _y_hat->getValue().rows() << ", " << _y_hat->getValue().cols()<<  std::endl;
		_diff = _y->getValue() - _y_hat->getValue();
		//std::cout << "MSE: " << _diff.rows() << ", " << _diff.cols() << std::endl;
		auto diff2= _diff.cwiseProduct(_diff);
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


	std::vector<Node*> topological_sort(Node* input_nodes)
	{
		//根据传入的数据初始化图结构
		Node* pNode = nullptr;
		//pair第一个为输入,第二个为输出
		std::map < Node*, std::pair<std::set<Node*>, std::set<Node*> > > g;
		//待遍历的周围节点
		std::list<Node*> vNodes;
		vNodes.emplace_back(input_nodes);
		//广度遍历,先遍历输出节点,再遍历输入节点
		//已经遍历过的节点
		std::set<Node*> sVisited;
		while (vNodes.size() && (pNode = vNodes.front())) {
			if (sVisited.find(pNode) != sVisited.end()) vNodes.pop_front();
			const auto& outputs = pNode->getOutputs();
			for (auto item: outputs)
			{
				g[pNode].second.insert(item);	//添加item为pnode的输出节点
				g[item].first.insert(pNode);	//添加pnode为item的输入节点
				if(sVisited.find(item)==sVisited.end()) vNodes.emplace_back(item);	//把没有访问过的节点添加到待访问队列中
			}
			const auto& inputs = pNode->getInputs();
			for (auto item: inputs)
			{
				g[pNode].first.insert(item);	//添加item为pnode的输入节点
				g[item].second.insert(pNode);	//添加pnode为item的输出节点
				if (sVisited.find(item) == sVisited.end()) vNodes.emplace_back(item);
			}
			//std::cout << pNode->name() << std::endl;
			sVisited.emplace(pNode);
			vNodes.pop_front();
		}

		//根据图结构进行拓扑排序
		std::vector<Node*> vSorted;
		while (g.size()) {
			for (auto itr=g.begin();itr!=g.end();++itr)
			{
				//没有输入节点
				auto& f = g[itr->first];
				if (f.first.size() == 0) {
					vSorted.push_back(itr->first);
					//找到图中这个节点的输出节点，然后将输出节点对应的这个父节点移除
					auto outputs = f.second;//f['out']
					for (auto& output: outputs)
					{
						g[output].first.erase(itr->first);
					}
					//然后将这个节点从图中移除
					g.erase(itr->first);
					break;
				}
			}
		}
		return vSorted;
	}

	DLL_EXPORT void train_one_batch(std::vector<Node*>& graph)
	{
		for (auto node:graph)
		{
			node->forward();
		}
		for (int idx = graph.size() - 1; idx >= 0;--idx) {
			graph[idx]->backward();
		}
	}

	DLL_EXPORT void sgd_update(std::vector<Node*> update_nodes, float learning_rate)
	{
		for (auto node: update_nodes)
		{
			Eigen::MatrixXf delta = -1 * learning_rate * node->getGradient(node);
			//std::cout << node->name()<<": "<< node->getValue().rows()<<", "<< node->getValue().cols()
			//	<< ", Delta: " << delta.rows()<<", "<< delta.cols() << std::endl;
			//Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
			//std::cout <<"node "<<node->name()<<", Delta: \n"<<delta.format(HeavyFmt) << std::endl;
			node->setValue(node->getValue() + delta);
		}
	}

}