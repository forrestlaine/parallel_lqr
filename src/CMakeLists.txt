cmake_minimum_required(VERSION 3.10)
project(motorboat)

enable_testing()

find_package(OpenMP REQUIRED)
#find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(LAPACK)

set(CMAKE_CXX_STANDARD 11)

INCLUDE_DIRECTORIES( "$ENV{EIGEN3_INCLUDE_DIR} ")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

if (DEBUG)
    message("Debug is on")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb3")
endif (DEBUG)

set(CMAKE_CXX_FLAGS "-O3 -fopenmp -Wall -Wextra -llapack -lblas")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")

add_library(motorboat SHARED trajectory.cpp trajectory.h parent_trajectory.h parent_trajectory.cpp dynamics.h dynamics.cpp running_constraint.cpp running_constraint.h numerical_gradient.h numerical_gradient.cpp endpoint_constraint.cpp endpoint_constraint.h running_cost.cpp running_cost.h terminal_cost.cpp terminal_cost.h)
add_executable(test_trajectory test_trajectory.cpp trajectory.cpp trajectory.h parent_trajectory.h parent_trajectory.cpp dynamics.h dynamics.cpp running_constraint.cpp running_constraint.h numerical_gradient.h numerical_gradient.cpp endpoint_constraint.cpp endpoint_constraint.h running_cost.cpp running_cost.h terminal_cost.cpp terminal_cost.h)
#target_link_libraries(motorboat Eigen3::Eigen)
target_link_libraries(test_trajectory)

