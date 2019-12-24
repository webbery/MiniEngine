#include "Util/Dataset.h"
#include <random>
#include <iostream>

namespace engine {
	namespace detail {

	}

	std::vector<Eigen::MatrixXf> make_blob(size_t samples, size_t features /*= 2*/, size_t centers /*= 3*/, float* pbox/*=10.0*/, float* cstd/*=1.0*/)
	{
		std::random_device rd{};
		std::mt19937 gen{ rd() };
		float* pStd = cstd;
		if (pStd == nullptr) {
			pStd = new float[centers];
			for (int i=0;i<centers;++i) pStd[i] = 1.0;
		}
		std::vector<std::vector<float>> vCenters;
		if (pbox == nullptr) {
			float boxes[2] = { -10, 10 };
			for (int g=0;g<centers;++g)
			{
				std::vector<float> vCenter;
				std::uniform_real_distribution<> ui(boxes[0], boxes[1]);
				for (int f = 0; f < features; ++f) {
					double number = ui(gen);
					vCenter.emplace_back(number);
				}
				vCenters.emplace_back(vCenter);
			}
		}
		//else {
		//	for (int g = 0; g < centers; ++g)
		//	{
		//		for (int f = 0; f < features; ++f) {
		//			double number =pbox[;
		//			pMean[g + f * features] = number;
		//		}
		//	}
		//}
		std::vector<Eigen::MatrixXf> vSamples;
		
		for (size_t g = 0; g < centers; ++g) {
			Eigen::MatrixXf blob(samples, features);
			std::cout << vCenters[g][0] <<", "<< vCenters[g][1]<< std::endl;
			for (size_t f = 0; f < features; ++f) {
				std::normal_distribution<double> distribution(vCenters[g][f], pStd[g]);
				for (int i = 0; i < samples; ++i) {
					double number = distribution(gen)/* + vCenters[g][f]*/;
					blob(i, f) = number;
				}
			}
			vSamples.emplace_back(blob);
		}
		
		if (cstd == nullptr) {
			delete[] pStd;
		}
		return std::move(vSamples);
	}

}