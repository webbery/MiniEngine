#pragma once 
#include "Node.h"

namespace engine {
	class RNN : public Node {
	public:
		RNN(Node* input, Node* weights, Node* bias);
	private:
	};
}