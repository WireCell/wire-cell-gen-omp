#!/bin/bash

echo "set ENVs"

export WC_GEN_OMP_SRC=/home/twang/myWork/wire-cell-gen-omp-cpu

#WC_GEN_OMP build directory 
[ -z $WC_GEN_OMP_BUILD ] && export WC_GEN_OMP_BUILD=$PWD
if [ $WC_GEN_OMP_BUILD == $WC_GEN_OMP_SRC ] ; then
	export WC_GEN_OMP_BUILD=${WC_GEN_OMP_SRC}/build
fi

alias wc_run="lar -n 1 -c ${WC_GEN_OMP_SRC}/example/sim.fcl  ${WC_GEN_OMP_SRC}/example/g4-1-event.root"
#alias wcb_configure="${WC_GEN_OMP_SRC}/configure.out  ${OPENMP_PATH} ${WC_GEN_OMP_BUILD} "
#alias wcb_build="${WC_GEN_OMP_SRC}/wcb -o $WC_GEN_OMP_BUILD -t ${WC_GEN_OMP_SRC} build --notest "

#alias wc-build-cmake="cmake ${WC_GEN_OMP_SRC}/.cmake-omp-cuda/ && make"
#alias wc-build-cmake-hip="${WC_GEN_OMP_SRC}/build-cmake.hip"

#echo "WC_OMP_GEN_BUILD directory: $WC_GEN_OMP_BUILD "
echo "WC_GEN_OMP_SRC directory: ${WC_GEN_OMP_SRC}"
#alias

#no-container
export WIRECELL_DATA=/home/zdong/PPS/git/wire-cell-data

export WC_GEN_OMP_LIB=$WC_GEN_OMP_BUILD

export WCT=$WIRECELL_FQ_DIR
export WCT_SRC=${WIRECELL_FQ_DIR}/wirecell-0.14.0

export WIRECELL_PATH=${WC_GEN_OMP}:${WC_GEN_OMP_SRC}/cfg:${WC_GEN_OMP_SRC}/example:${WIRECELL_DATA}:${WCT_SRC}/cfg:${WCT}/share/wirecell
export LD_LIBRARY_PATH=${WC_GEN_OMP_LIB}:${CUDA_DIR}/lib64:${WCT}/lib:$LD_LIBRARY_PATH
