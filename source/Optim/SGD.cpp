#include "Optim/SGD.h"

namespace engine {
	DLL_EXPORT void sgd_update(std::vector<Node*> update_nodes, float learning_rate)
	{
		for (auto node : update_nodes)
		{
			Eigen::MatrixXf delta = -1 * learning_rate * node->getGradient(node);
			//std::cout << node->name()<<": "<< node->getValue().rows()<<", "<< node->getValue().cols()
			//	<< ", Delta: " << delta.rows()<<", "<< delta.cols() << std::endl;
			//Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
			//std::cout <<"node "<<node->name()<<", Delta: \n"<<delta.format(HeavyFmt) << std::endl;
			node->setValue(node->getValue() + delta);
		}
	}

	
	void SGD::update(std::vector<Node*> update_nodes, float learning_rate)
	{
		for (auto node : update_nodes)
		{
			node->update([&](Node* pNode)->Node * {
				Eigen::MatrixXf delta = -1 * learning_rate * pNode->getGradient(pNode);
				pNode->setValue(pNode->getValue() + delta);
				return pNode;
			});
		}
	}

	void SGD::minimize(Loss* loss)
	{

	}

}