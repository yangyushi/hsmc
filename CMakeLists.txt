cmake_minimum_required(VERSION 3.15)

# set the project name
project(hsmc VERSION 0.0.0)

include(GNUInstallDirs)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
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

# install the Python package and its extension module into the wheel/site-packages
install(
  DIRECTORY lib/hsmc/
  DESTINATION hsmc
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
install(
  DIRECTORY workflow/
  DESTINATION hsmc/workflow
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
install(
  TARGETS chard_sphere
  LIBRARY DESTINATION hsmc
  RUNTIME DESTINATION hsmc
)
