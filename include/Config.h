#pragma once
#ifdef __USE_OPENCL__
#include <boost/compute.hpp>
#include <boost/compute/interop/eigen.hpp>
#else
#include <Eigen/Core>
#endif

#ifdef MiniEngine_EXPORTS
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __declspec(dllimport)
#endif

#pragma warning(disable:4251)
