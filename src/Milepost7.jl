#############################################################################
# ProxAL/ExaTron Example File
# This example runs ProxAL with ExaTron as a backend and outputs a profile
# file. PProf has to be installed in the global environment.
#############################################################################

module Milepost7
using ArgParse
using CUDA
# using DelimitedFiles, Printf
using LinearAlgebra, JuMP, Ipopt
using AMDGPU
# using oneAPI
using MPI
using Profile
using PProf
using ProxAL
using Logging

export milepost7, parse_cmd

function milepost7(
    case::String,
    demandfiles::String,
    T::Int64,
    K::Int64,
    configuration::Int64;
    profile=false,
    output=false,
    rhopq=3e3,
    rhova=3e4,
    proxal_iter=100,
    exatron_inner_iter=400,
    exatron_outer_iter=1
)
    # Load case
    # const DATA_DIR = "cases/n-2"
    # case_file = joinpath(DATA_DIR, "$(case)")
    # load_file = joinpath(DATA_DIR, demandfiles)
    case_file = case
    load_file = demandfiles

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
    algparams.iterlim = proxal_iter
    algparams.verbose_inner = 0

    # choose backend
    if configuration == 1
        backend = ProxAL.JuMPBackend()
    elseif 2 <= configuration <= 4
        backend = ProxAL.AdmmBackend()
    else
        error("Unsupported configuration $(configuration)")
    end
    if configuration == 2
        algparams.ka_device = nothing
    elseif configuration == 3
        algparams.device = ProxAL.GPU
        algparams.ka_device = nothing
    elseif configuration == 4
        algparams.device = ProxAL.KADevice
        if CUDA.has_cuda_gpu()
            gpu_device = CUDABackend()
        elseif AMDGPU.has_rocm_gpu()
            gpu_device = ROCBackend()
        end
        algparams.ka_device = gpu_device
    else
        error("Configuration error $(configuration)")
    end
    algparams.optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0) #,  "tol" => 1e-1*algparams.tol)
    algparams.tron_rho_pq=rhopq
    algparams.tron_rho_va=rhova
    algparams.tron_outer_iterlim=exatron_outer_iter
    algparams.tron_inner_iterlim=exatron_inner_iter
    algparams.mode = :coldstart
    algparams.init_opf = false
    # algparams.tron_outer_eps = Inf


    ranks = MPI.Comm_size(MPI.COMM_WORLD)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println("ProxAL/ExaTron $ranks ranks, $T periods, $K contingencies")
        println("Case: $case")
        println("tron_inner_iterlim: $(algparams.tron_inner_iterlim)")
        println("tron_outer_iterlim: $(algparams.tron_outer_iterlim)")
        println("tron_rho_va: $(algparams.tron_rho_va)")
        println("tron_rho_pq: $(algparams.tron_rho_pq)")
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
            output=output
        )
        # end
    end
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        # global_logger(cur_logger)
        println("Creating problem: $elapsed_t")
        println("Solver start")
        np = MPI.Comm_size(MPI.COMM_WORLD)
        if !profile
            println("No profiling")
            elapsed_t = @elapsed begin
                info = ProxAL.optimize!(nlp)
            end
        else
            println("Profiling")
            info = ProxAL.optimize!(nlp)
            Profile.clear()
            elapsed_t = @elapsed begin
                Profile.@profile begin
                    info = ProxAL.optimize!(nlp)
                end
            end
            PProf.pprof()
        end
        println("AugLag iterations: $(info.iter) with $np ranks in $elapsed_t seconds")
    else
        info = ProxAL.optimize!(nlp)
        if profile
            info = ProxAL.optimize!(nlp)
        end
    end
    return info
end

function parse_cmd()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--profile"
            help = "Create a plot of the load"
            action = :store_true
        "--output"
            help = "Write output using HDF5.jl and MPI if available"
            action = :store_true
        "case"
            help = "Case file"
            arg_type = String
            required = true
        "load"
            help = "Load files"
            arg_type = String
            required = true
        "T"
            help = "Number of time periods"
            arg_type = Int64
            required = true
        "K"
            help = "Number of contingencies"
            arg_type = Int64
            required = true
        "--rhopq"
            help = "rho Pq parameter in ExaTron"
            arg_type = Float64
            default = 3e3
        "--rhova"
            help = "rho Va parameter in ExaTron"
            arg_type = Float64
            default = 3e4
        "--configuration"
            help = "Configuration to run on\n 1: JuMP/Ipopt\n 2: ExaAdmm/CPU\n 3: ExaAdmm/CUDA\n 4: ExaAdmm/KernelAbstractions"
            arg_type = Int64
            default = 1
        "--proxal_iter"
            help = "Maximum number of iterations for ProxAL"
            arg_type = Int64
            default = 100
        "--exatron_inner_iter"
            help = "Maximum number of iterations for ExaTron inner loop"
            arg_type = Int64
            default = 400
        "--exatron_outer_iter"
            help = "Maximum number of iterations for ExaTron outer loop"
            arg_type = Int64
            default = 1
    end

    return parse_args(s)
end

function julia_main()::Cint
    MPI.Init()

    if MPI.Comm_rank(MPI.COMM_WORLD) != 0
        disable_logging(Info)
        disable_logging(Warn)
    else
        if AMDGPU.has_rocm_gpu()
            @show AMDGPU.libhsaruntime_path
            @show AMDGPU.use_artifacts
        end
    end

    if isinteractive() && length(ARGS) == 0
        case = "/lustre/orion/csc359/scratch/mschanen/git/milepost7/cases/case_ACTIVSg10k.m"
        load = "/lustre/orion/csc359/scratch/mschanen/git/milepost7/cases/case_ACTIVSg10k"
        profile = false
        output = true
        T = 2
        K = 1
        rhopq = 3e3
        rhova = 3e4
        configuration = 4
        exatron_inner_iter = 400
        exatron_outer_iter = 1
        proxal_iter = 2
    else
        args = parse_cmd()
        profile = args["profile"]
        output = args["output"]
        case = args["case"]
        load = args["load"]
        T = args["T"]
        K = args["K"]
        rhopq = args["rhopq"]
        rhova = args["rhova"]
        configuration = args["configuration"]
        proxal_iter = args["proxal_iter"]
        exatron_inner_iter = args["exatron_inner_iter"]
        exatron_outer_iter = args["exatron_outer_iter"]
    end

    info = milepost7(
        case, load, T, K, configuration;
        profile=profile,
        output=output,
        rhopq=rhopq, rhova=rhova,
        proxal_iter=proxal_iter,
        exatron_inner_iter=exatron_inner_iter,
        exatron_outer_iter=exatron_outer_iter
    );

    # do something based on ARGS?
    return 0 # if things finished successfully
    end

end
