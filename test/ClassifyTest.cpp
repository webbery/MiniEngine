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

int main() {
	using namespace  std;
	//使用随机数生成三类聚类数据
	using namespace engine;
	auto blobs = make_blob(100);
	std::ofstream ofs;
	ofs.open("blob.csv", std::ios::out | std::ios::ate);
	for (auto group : blobs)
	{
		size_t rows = group.rows();
		size_t cols = group.cols();
		for (size_t r = 0; r < rows; ++r) {
			for (size_t c = 0; c < cols; ++c) {
				auto val = std::to_string(group(r, c));
				ofs.write(val.c_str(), val.size());
				ofs.write(",", 1);
			}
			ofs.write("\n", 1);
		}
	}
	ofs.close();
	//Init
	
}