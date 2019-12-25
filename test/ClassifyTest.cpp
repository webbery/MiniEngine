#include "MiniEngine.h"
#include "F/Softmax.h"
#include "Layer/Linear.h"
#include "Loss/MSE.h"
#include "Layer/Dropout.h"
#include "Optim/SGD.h"
#include "Util/Dataset.h"
#include <iostream>
#include <algorithm>
#include <fstream>

#define SAMPLE_SIZE	100
#define SAMPLE_FEATURE	2
int main() {
	using namespace  std;
	//使用随机数生成三类聚类数据
	using namespace engine;
	auto blobs = make_blob(SAMPLE_SIZE);
	Eigen::MatrixXf data(blobs.size()* SAMPLE_SIZE, SAMPLE_FEATURE);
	Eigen::MatrixXf labels=Eigen::MatrixXf::Zero(blobs.size() * SAMPLE_SIZE, 3);
	for (auto idx=0;idx<blobs.size();++idx)
	{
		data.block<SAMPLE_SIZE, SAMPLE_FEATURE>(idx * SAMPLE_SIZE, 0) = blobs[idx];
		labels.block<SAMPLE_SIZE, 1>(idx * SAMPLE_SIZE, idx) = Eigen::MatrixXf::Ones(SAMPLE_SIZE, 1);
	}
	auto samples = resample(data, labels);
	//std::ofstream ofs;
	//ofs.open("blob.csv", std::ios::out | std::ios::ate);
	//for (auto group : blobs)
	//{
	//	size_t rows = group.rows();
	//	size_t cols = group.cols();
	//	for (size_t r = 0; r < rows; ++r) {
	//		for (size_t c = 0; c < cols; ++c) {
	//			auto val = std::to_string(group(r, c));
	//			ofs.write(val.c_str(), val.size());
	//			ofs.write(",", 1);
	//		}
	//		ofs.write("\n", 1);
	//	}
	//}
	//ofs.close();

#define HIDDEN_SIZE 10
	//构造分类器
	Input X("X"), y("y");
	Input W1("W1", SAMPLE_FEATURE, HIDDEN_SIZE), b1("b1", HIDDEN_SIZE, 3);
	auto linear1 = Linear(&X, &W1, &b1);
	auto y_hat = Softmax(&linear1);
	auto loss = MSE(&y, &y_hat);
	auto g = make_graph(&X);
	SGD sgd;

	int epochs = 1000;
	int batch_size = 16;
	int steps_per_epoch = SAMPLE_SIZE / batch_size;
	for (auto epoch = 0; epoch < epochs; ++epoch) {
		loss.reset();
		for (size_t batch = 0; batch < steps_per_epoch; ++batch) {
			auto batch_x = samples.first.block(batch * batch_size, 0, batch_size, SAMPLE_FEATURE);
			auto batch_y = samples.second.block(batch * batch_size, 0, batch_size, 3);
			X.setValue(batch_x);
			y.setValue(batch_y);

			train_one_batch(g);
			sgd.update(g, 1e-4);

			//std::cout << graph[graph.size() - 1]->getValue().rows() << "," << graph[graph.size() - 1]->getValue().cols() << std::endl;
			loss += g[g.size() - 1]->getValue();
		}
	}
}