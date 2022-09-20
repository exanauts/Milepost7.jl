#!/bin/sh
#PBS -l select=20:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
##PBS -q debug 
#PBS -A CSC249ADSE22 

cd ${PBS_O_WORKDIR}

# MPI example w/ 4 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NDEPTH=8
NTHREADS=1
module load cray-hdf5-parallel 
export TMPDIR=$PWD/TMPDIR
export JULIA_DEPOT_PATH=/lus/grand/projects/ExaSGD/mschanen/julia-depot-polaris
export JULIA_BIN=/lus/grand/projects/ExaSGD/mschanen/julia/julia-1.8.1/bin/julia
export JULIA_CUDA_USE_BINARYBUILDER=false

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE=${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth ${JULIA_BIN} --check-bounds=no --project milepost7_20.jl
