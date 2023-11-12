cmake_minimum_required(VERSION 3.12)

project(oetsort LANGUAGES CUDA CXX)

find_package(caliper REQUIRED)
find_package(adiak REQUIRED)


add_executable(oetsort_cuda oetsort_cuda.cu)

message(STATUS "CUDA includes : ${CUDA_INCLUDE_DIRS}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})

target_link_libraries(oetsort_cuda PRIVATE caliper)