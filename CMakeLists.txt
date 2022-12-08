cmake_minimum_required(VERSION 3.5)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

project(open3d_learning)

set(CMAKE_CXX_STANDARD 17)

find_package(Open3D REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(OpenCV REQUIRED)

add_executable(tsdf tsdf.cpp)
#target_link_directories(open3d_grid PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tsdf Open3D::Open3D Eigen3::Eigen)

add_executable(custom_tsdf custom_tsdf.cpp)
#target_link_directories(open3d_grid PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(custom_tsdf Open3D::Open3D Eigen3::Eigen)
