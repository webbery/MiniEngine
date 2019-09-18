#include "MiniEngine.h"
#ifdef __USE_OPENCL__
#include <boost/compute.hpp>
#include <boost/compute/interop/eigen.hpp>
#else
#include <Eigen/Core>
#endif
#include <iostream>

int main() {
	//using namespace  std;
	//Eigen::MatrixXf X(1, 3);
	//X << 1, 2, 3;
	//cout << X << endl;
	//Eigen::MatrixXf W(3, 2);
	//W << 1, 2, 2, 4, 3, 5;
	//cout << W << endl;
	//Eigen::MatrixXf B(1, 2);
	//B << 1, 3;

	//cout << X * W + B << endl;
}