cmake_minimum_required (VERSION 3.6)

project(MiniEngineTest)

include_directories(../include ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

link_directories(E:/code/C++/Develop/boost_1_68_0/stage/lib)


set(TEST_SOURCE EngineTest.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE})

