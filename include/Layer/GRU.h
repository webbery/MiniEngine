#pragma once
#include "RNN.h"

namespace engine {
	class DLL_EXPORT GRU final: public Node {
	public:
		GRU(Node* input);

		void forward();
		void backward();

	private:

	};
}