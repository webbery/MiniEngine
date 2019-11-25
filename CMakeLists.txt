cmake_minimum_required (VERSION 3.6)

project(MiniEngine)

aux_source_directory(source EningeSource)
aux_source_directory(source/F FunctionSource)
aux_source_directory(source/Layer LayerSource)
aux_source_directory(source/Loss LossSource)

set( BOOST_COMPONENTS_NEEDED compute )

# The following verifyies that BOOST_ROOT is set properly.
if(NOT BOOST_ROOT AND NOT $ENV{BOOST_ROOT} STREQUAL "")
    FILE( TO_CMAKE_PATH $ENV{BOOST_ROOT} BOOST_ROOT )
    if( NOT EXISTS ${BOOST_ROOT} )
        MESSAGE( STATUS  ${BOOST_ROOT} " does not exist. Checking if BOOST_ROOT was a quoted string.." )
        STRING( REPLACE "\"" "" BOOST_ROOT ${BOOST_ROOT} )
        if( EXISTS ${BOOST_ROOT} )
            MESSAGE( STATUS "After removing the quotes " ${BOOST_ROOT} " was now found by CMake" )
        endif( EXISTS ${BOOST_ROOT})
    endif( NOT EXISTS ${BOOST_ROOT} )

# Save the BOOST_ROOT in the cache
    if( NOT EXISTS ${BOOST_ROOT} )
        MESSAGE( WARNING ${BOOST_ROOT} " does not exist." )
    else(NOT EXISTS ${BOOST_ROOT})
        SET (BOOST_ROOT ${BOOST_ROOT} CACHE STRING "Set the value of BOOST_ROOT to point to the root folder of your boost install." FORCE)
        #SET (BOOST_INCLUDEDIR ${BOOST_ROOT}/Include)
        #SET (BOOST_LIBRARYDIR ${BOOST_ROOT}/lib)
    endif( NOT EXISTS ${BOOST_ROOT} )

endif(NOT BOOST_ROOT AND NOT $ENV{BOOST_ROOT} STREQUAL "")

if( WIN32 AND NOT BOOST_ROOT )
    MESSAGE( WARNING "Please set the BOOST_ROOT environment variable." )
endif( WIN32 AND NOT BOOST_ROOT )

set(Boost_ADDITIONAL_VERSIONS "1.68" "1.68.0")
set(Boost_DEBUG ON)
set(Boost_USE_STATIC_LIBS       OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
FIND_PACKAGE(Boost 1.68.0 COMPONENTS ${BOOST_COMPONENTS_NEEDED})
if(Boost_FOUND)
    MESSAGE( STATUS "Setting up boost." )
    include_directories(${Boost_INCLUDE_DIRS})
    if(Boost_DEBUG)
        MESSAGE( STATUS "BOOST Libraries " ${Boost_LIBRARIES} )
        FOREACH(BOOST_COMPONENT ${BOOST_COMPONENTS_NEEDED})
            STRING( TOUPPER ${BOOST_COMPONENT} BOOST_COMPONENT_UPCASE )
            MESSAGE( STATUS "Boost " ${BOOST_COMPONENT} ": " ${Boost_${BOOST_COMPONENT_UPCASE}_LIBRARY} )
            MESSAGE( STATUS "Boost " ${BOOST_COMPONENT} " Debug: " ${Boost_${BOOST_COMPONENT_UPCASE}_LIBRARY_DEBUG} )
            MESSAGE( STATUS "Boost " ${BOOST_COMPONENT} " Release: " ${Boost_${BOOST_COMPONENT_UPCASE}_LIBRARY_RELEASE} )
        ENDFOREACH(BOOST_COMPONENT)
    endif(Boost_DEBUG)
endif(Boost_FOUND)

if(Boost_FOUND)
find_package(OpenCL)
if(OpenCL_FOUND)
message(Use_OpenCL: ${OpenCL_VERSION_STRING})
add_definitions(-D__USE_OPENCL__)
include_directories(${OpenCL_INCLUDE_DIR})
endif(OpenCL_FOUND)

include_directories(${Boost_INCLUDE_DIRS})

## compiler feature

else(Boost_FOUND)

endif(Boost_FOUND)

set(CMAKE_CXX_STANDARD 17)

# find_path(Boost_INCLUDE_DIR boost/any.hpp PATHS /usr/local/include)
# link_directories(E:/code/C++/Develop/boost_1_68_0/stage/lib ${Boost_LIBRARIES_DIR})

find_package(Eigen3)
message(${EIGEN3_INCLUDE_DIR})
include_directories(include ${EIGEN3_INCLUDE_DIR})

add_library(${PROJECT_NAME} SHARED ${EningeSource} ${FunctionSource} ${LayerSource} ${LossSource})

if(Boost_FOUND)
message(boost_dir:${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif(Boost_FOUND)
if(OpenCL_FOUND)
target_link_libraries( ${PROJECT_NAME} ${OpenCL_LIBRARIES})
endif(OpenCL_FOUND)
add_subdirectory(test)