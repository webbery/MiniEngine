#pragma once

namespace engine {
	//class Complex {

	//};

	class Img {
	public:
		Img(double i) :_i(i) {}
	private:
		double _i=0;
	};

	Img operator"" j(long double x) { return x; }
}