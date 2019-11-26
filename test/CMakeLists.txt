cmake_minimum_required (VERSION 3.6)

project(MiniEngineTest)

#if(Boost_FOUND)
#include_directories(${Boost_INCLUDE_DIRS})
#link_directories(E:/code/C++/Develop/boost_1_68_0/stage/lib)
#endif(Boost_FOUND)
include_directories(../include ${EIGEN3_INCLUDE_DIR})

add_executable(mini_demo EngineTest.cpp)
target_link_libraries(mini_demo MiniEngine)

add_executable(test_rnn RNNTest.cpp)
target_link_libraries(test_rnn MiniEngine)

