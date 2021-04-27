cmake_minimum_required(VERSION 3.10)

# set the project name
project(hsmc VERSION 0.0)
find_package(pybind11 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

# setup usual compile flag
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# add the executable
add_library(hard_sphere STATIC src/hard_sphere.cpp)
target_link_libraries(hard_sphere Eigen3::Eigen)

# build the python module
add_library(chard_sphere MODULE src/chard_sphere.cpp)
target_link_libraries(chard_sphere PRIVATE pybind11::module pybind11::lto Eigen3::Eigen hard_sphere)
pybind11_extension(chard_sphere)
set_target_properties(chard_sphere PROPERTIES CXX_VISIBILITY_PRESET "hidden")

# move the libraries to lib folder
SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/lib/hsmc)
install(TARGETS chard_sphere DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS hard_sphere DESTINATION ${CMAKE_SOURCE_DIR}/lib)
