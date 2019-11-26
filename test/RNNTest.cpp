#include "MiniEngine.h"
#include "F/Sigmoid.h"
#include "Layer/Linear.h"
#include "Loss/MSE.h"
#include "Layer/GRU.h"
#include <iostream>

int main() {

	using namespace engine;
	Input X("X"), y("y");
	RNN y_hat(&X);
	auto mse = MSE(&y, &y_hat);
	auto graph = topological_sort(&X);

	for (auto epoch = 0; epoch < 2; ++epoch) {
		Eigen::MatrixXf loss(1, 1);
		loss.setZero();
		for (size_t batch = 0; batch < 2; ++batch) {
			//X.setValue(batch_x);
			//y.setValue(batch_y);

			//train_one_batch(graph);
			//sgd_update({ &W1, &W2, &b1, &b2 }, 1e-4);

			//std::cout << graph[graph.size() - 1]->getValue().rows() << "," << graph[graph.size() - 1]->getValue().cols() << std::endl;
			loss += graph[graph.size() - 1]->getValue();
		}
	}
}