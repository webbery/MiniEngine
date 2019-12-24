#include "Optim/SGD.h"

namespace engine {
	
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