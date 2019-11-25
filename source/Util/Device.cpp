#include "Util/Device.h"

namespace engine {

	Device::Device()
#ifdef __USE_OPENCL__
		:_gpu(compute::system::default_device())
#endif
	{

	}

}