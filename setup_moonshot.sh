#!/bin/bash
export JULIA_HDF5_PATH="/disk/hdf5/hdf5-1.12.2/build/bin"
export JULIA_MPI_BINARY="system"
export JULIA_MPI_PATH="/nfs/gce/software/custom/linux-ubuntu22.04-x86_64/spack/opt/spack/linux-ubuntu22.04-x86_64/gcc-11.3.0/openmpi-4.1.3-qrpnszy"
export OMP_NUM_THREADS=1
julia --project -e 'using Pkg; Pkg.build("MPI"; verbose=true); Pkg.build("HDF5"; verbose=true)'
julia --project -e 'using MPI ; MPI.install_mpiexecjl(force=true ;destdir=".")'
