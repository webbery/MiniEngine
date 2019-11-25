#pragma once
#ifdef __USE_OPENCL__
#include <boost/compute.hpp>
namespace compute = boost::compute;
#endif
namespace engine {
	class Device {
	public:
		Device();
	private:
#ifdef __USE_OPENCL__
		compute::device _gpu;
#else
#endif
	};
}