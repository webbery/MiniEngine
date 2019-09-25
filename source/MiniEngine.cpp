#include "MiniEngine.h"
#include <map>
#include <set>

namespace engine {

	Input::Input(const char* name/*, const std::vector<Node* >& inputs*/)
	{
		_name = name;
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
		_name = "Linear";
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
		_name = "Sigmoid";
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
		_name = "MSE";
	}

	void MSE::forward(const Eigen::MatrixXf& value)
	{
		_diff = _y->getValue() - _y_hat->getValue();
		auto v = Eigen::MatrixXf(1, 1);
		v<<(_diff * _diff).mean();
		_value = v;
	}

	void MSE::backward()
	{
		auto r = _y_hat->getValue().rows();
		_gradients[_y] = _diff * (2 / r);
		_gradients[_y_hat] = _diff * (-2 / r);
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
		}
	}

	DLL_EXPORT void sgd_update(Node* startNode)
	{

	}

}