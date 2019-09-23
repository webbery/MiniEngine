cmake_minimum_required (VERSION 3.6)

project(MiniEngineTest)

if(Boost_FOUND)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(E:/code/C++/Develop/boost_1_68_0/stage/lib)
endif(Boost_FOUND)
include_directories(../include ${EIGEN3_INCLUDE_DIR})

set(TEST_SOURCE EngineTest.cpp)

add_executable(${PROJECT_NAME} ${TEST_SOURCE})

target_link_libraries(${PROJECT_NAME} MiniEngine)
