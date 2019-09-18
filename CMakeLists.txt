cmake_minimum_required (VERSION 3.6)

project(MiniEngine)

aux_source_directory(source EningeSource)

find_package(Boost)
if(Boost_FOUND)

find_package(OpenCL)
if(OpenCL_FOUND)
add_definitions(-D__USE_OPENCL__)
include_directories(${OpenCL_INCLUDE_DIR})
endif(OpenCL_FOUND)

include_directories(${Boost_INCLUDE_DIR})

## compiler feature
set(CMAKE_CXX_STANDARD 17)

else(Boost_FOUND)

# find_path(Boost_INCLUDE_DIR boost/any.hpp PATHS /usr/local/include)
# link_directories(E:/code/C++/Develop/boost_1_68_0/stage/lib ${Boost_LIBRARIES_DIR})

endif(Boost_FOUND)

find_package(Eigen3)
message(${EIGEN3_INCLUDE_DIR})
include_directories(include ${EIGEN3_INCLUDE_DIR})

add_library(${PROJECT_NAME} SHARED ${EningeSource})

add_subdirectory(test)