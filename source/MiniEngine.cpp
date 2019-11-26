#include "MiniEngine.h"
#include <map>
#include <set>
#include <iostream>

namespace engine {
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

}