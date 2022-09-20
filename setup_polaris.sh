#!/bin/bash
module load cray-hdf5-parallel
export JULIA_HDF5_PATH=$HDF5_DIR
export JULIA_MPI_PATH=/opt/cray/pe/mpich/8.1.16/ofi/nvidia/20.7
julia --project -e 'ENV["JULIA_MPI_BINARY"]="system"; using Pkg; Pkg.build("MPI"; verbose=true); Pkg.build("HDF5"; verbose=true)'
julia --project -e 'using MPI ; MPI.install_mpiexecjl(force=true ;destdir=".")'
export OMP_NUM_THREADS=1
