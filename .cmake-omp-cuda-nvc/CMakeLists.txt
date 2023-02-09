cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(wire-cell-gen-openmp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS Off)
find_package(Boost REQUIRED COMPONENTS)
set(Boost_USE_MULTITHREADED ON)

file(GLOB all_files ${PROJECT_SOURCE_DIR}/../src/*.cxx)

#find_package(CUDA REQUIRED)

string(APPEND CMAKE_CXX_FLAGS "-g --pedantic -mp=gpu -O3 -cudalib=cublas,curand,cufft -DOPENMP_ENABLE_CUDA -DARCH_CUDA")
string(APPEND CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

add_library(WireCellGenOpenMP SHARED ${all_files})
target_include_directories(WireCellGenOpenMP
  PRIVATE
    ${PROJECT_SOURCE_DIR}/../inc
    $ENV{OMPRNG}
    $ENV{EIGEN_INC}
    $ENV{JSONCPP_INC}
    $ENV{JSONNET_INC}
    $ENV{SPDLOG_INC}
    $ENV{WIRECELL_INC}
)
set_target_properties(WireCellGenOpenMP
	PROPERTIES COMPILE_OPTIONS "-DEIGEN_NO_CUDA;-DEIGEN_DONT_VECTORIZE")

target_link_directories(WireCellGenOpenMP PRIVATE  $ENV{JSONCPP_LIB} $ENV{WIRECELL_LIB})
target_link_libraries(WireCellGenOpenMP PRIVATE  jsoncpp WireCellIface WireCellUtil)

#add_subdirectory(test)

#message(STATUS "CUDA_cufft_LIBRARY: ${CUDA_cufft_LIBRARY}")
#message(STATUS "CUDA_curand_LIBRARY: ${CUDA_curand_LIBRARY}")
