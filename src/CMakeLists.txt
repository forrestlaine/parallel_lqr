cmake_minimum_required(VERSION 2.8)
project(motorboat)

enable_testing()

find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#find_package(LAPACK)

add_compile_options(-std=c++11)
#set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (DEBUG)
    message("Debug is on")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb3")
endif (DEBUG)

set(CMAKE_CXX_FLAGS "-O3 -fopenmp -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")

add_library(motorboat SHARED trajectory.cpp trajectory.h parent_trajectory.h parent_trajectory.cpp dynamics.h dynamics.cpp running_constraint.cpp running_constraint.h numerical_gradient.h numerical_gradient.cpp endpoint_constraint.cpp endpoint_constraint.h running_cost.cpp running_cost.h terminal_cost.cpp terminal_cost.h)
add_executable(test_trajectory test_trajectory.cpp trajectory.cpp trajectory.h parent_trajectory.h parent_trajectory.cpp dynamics.h dynamics.cpp running_constraint.cpp running_constraint.h numerical_gradient.h numerical_gradient.cpp endpoint_constraint.cpp endpoint_constraint.h running_cost.cpp running_cost.h terminal_cost.cpp terminal_cost.h)