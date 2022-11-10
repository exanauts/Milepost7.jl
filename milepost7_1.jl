#############################################################################
# ProxAL/ExaTron Example File
# This example runs ProxAL with ExaTron as a backend and outputs a profile
# file. PProf has to be installed in the global environment.
#############################################################################

using Infiltrator
using KernelAbstractions
using ProxAL
# using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using CUDA
using AMDGPU
using MPI
const KA = KernelAbstractions
using CUDAKernels
using ROCKernels

function ProxAL.ExaAdmm.KAArray{T}(n::Int, device::CUDADevice) where {T}
    return CuArray{T}(undef, n)
end

function ProxAL.ExaAdmm.KAArray{T}(n::Int, device::ROCDevice) where {T}
    return ROCArray{T}(undef, n)
end
MPI.Init()
@show AMDGPU.devices()


case = "case_ACTIVSg10k"
# case = "case9"
demandfiles = "$(case)"
# Load case
const DATA_DIR = "cases"
case_file = joinpath(DATA_DIR, "$(case).m")
load_file = joinpath(DATA_DIR, demandfiles)

# choose one of the following (K*T subproblems in each case)
if length(ARGS) == 0
    (T, K) = (16, 32)
    # (T, K) = (2, 1)
elseif length(ARGS) == 4
    case = ARGS[1]
    demandfiles = ARGS[2]
    T = parse(Int, ARGS[3])
    K = parse(Int, ARGS[4])
else
    println("Usage: [mpiexec -n nprocs] julia --project examples/exatron.jl [case demandfiles T K]")
    println("")
    println("       (case,demandfiles,T,K) defaults to (case9,case9_oneweek_168,2,1)")
    exit()
end

# choose backend
# backend = ProxAL.JuMPBackend()
# With ExaAdmmBackend(), CUDADevice will used
backend = ProxAL.AdmmBackend()

# Model/formulation settings
modelinfo = ModelInfo()
modelinfo.num_time_periods = T
modelinfo.load_scale = 1.0
modelinfo.ramp_scale = 0.2
modelinfo.corr_scale = 0.5
modelinfo.allow_obj_gencost = true
modelinfo.allow_constr_infeas = true
modelinfo.weight_constr_infeas = 1e1
modelinfo.time_link_constr_type = :penalty
modelinfo.ctgs_link_constr_type = :corrective_penalty
modelinfo.allow_line_limits = true
modelinfo.case_name = case
modelinfo.num_ctgs = K

# Algorithm settings
algparams = AlgParams()
algparams.verbose = 1
algparams.tol = 1e-3
algparams.decompCtgs = (K > 0)
algparams.iterlim = 500
if isa(backend, ProxAL.AdmmBackend)
    # algparams.device = ProxAL.CUDADevice
    algparams.device = ProxAL.KADevice
    algparams.ka_device = ROCDevice()
end
algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0) #,  "tol" => 1e-1*algparams.tol)
algparams.tron_rho_pq=3e3
algparams.tron_rho_pa=3e4
algparams.tron_outer_iterlim=8
algparams.tron_inner_iterlim=250
algparams.mode = :coldstart
algparams.init_opf = false
algparams.tron_outer_eps = Inf


ranks = MPI.Comm_size(MPI.COMM_WORLD)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("ProxAL/ExaTron $ranks ranks, $T periods, $K contingencies")
end
# cur_logger = global_logger(NullLogger())
elapsed_t = @elapsed begin
    # redirect_stdout(devnull) do
        global nlp = ProxALEvaluator(
        case_file,
        load_file,
        modelinfo,
        algparams,
        backend;
        output=true
    )
    # end
end
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    # global_logger(cur_logger)
    println("Creating problem: $elapsed_t")
    println("Benchmark Start")
    np = MPI.Comm_size(MPI.COMM_WORLD)
    elapsed_t = @elapsed begin
        info = ProxAL.optimize!(nlp)
    end
    println("AugLag iterations: $(info.iter) with $np ranks in $elapsed_t seconds")
else
    info = ProxAL.optimize!(nlp)
end

if !isinteractive()
    MPI.Finalize()
end
#   [052768ef] CUDA v3.12.0
#   [72cfdca4] CUDAKernels v0.4.3
#   [f67ccb44] HDF5 v0.16.11
#   [b6b21f68] Ipopt v1.1.0
#   [4076af6c] JuMP v1.3.0
#   [63c18a36] KernelAbstractions v0.8.4
#   [da04e1cc] MPI v0.19.2
#   [91a5bcdd] Plots v1.34.1
#   [12c3852d] ProxAL v0.6.0 `/disk/mschanen/julia_depot/dev/ProxAL`
#   [7eb9e9f0] ROCKernels v0.3.2
#   [8bb1440f] DelimitedFiles
