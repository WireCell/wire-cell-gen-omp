cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(wire-cell-gen-openmp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS Off)

file(GLOB all_files ${PROJECT_SOURCE_DIR}/../src/*.cxx)

include_directories(SYSTEM $ENV{BOOST_INC})

set(BOOST_ROOT $ENV{BOOST_DIR} )
set(BOOST_INCLUDEDIR $ENV{BOOST_INC})
set(BOOST_LIBRARYDIR $ENV{BOOST_LIB})
set(Boost_USE_MULTITHREADED ON)

find_package(Boost REQUIRED COMPONENTS)

include_directories(Boost_INCLUDE_DIRS})


string(APPEND CMAKE_CXX_FLAGS "-g -O3 -pedantic -Wall -fopenmp -DUSE_RANDOM123")
string(APPEND CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

add_library(WireCellGenOpenMP SHARED ${all_files})
target_include_directories(WireCellGenOpenMP
  PRIVATE
    ${PROJECT_SOURCE_DIR}/../inc
    $ENV{OMPRNG}
    $ENV{FFTW_INC}
    $ENV{EIGEN_INC}
    $ENV{JSONCPP_INC}
    $ENV{JSONNET_INC}
    $ENV{SPDLOG_INC}
    $ENV{WIRECELL_INC}
)
set_target_properties(WireCellGenOpenMP
	PROPERTIES COMPILE_OPTIONS "-DEIGEN_NO_CUDA;-DEIGEN_FFTW_DEFAULT=1;-DEIGEN_DONT_VECTORIZE")

target_link_directories(WireCellGenOpenMP PRIVATE  $ENV{JSONCPP_LIB} $ENV{WIRECELL_LIB})
target_link_libraries(WireCellGenOpenMP PRIVATE  jsoncpp WireCellIface WireCellUtil Boost::headers)

#add_subdirectory(test)
