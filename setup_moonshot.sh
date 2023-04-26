#!/bin/bash
export JULIA_HDF5_PATH="/disk/hdf5/hdf5-1.12.2/build/bin"
export OMP_NUM_THREADS=1
julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
julia --project -e 'using Pkg; Pkg.build("MPI"; verbose=true); Pkg.build("HDF5"; verbose=true)'
