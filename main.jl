using AMDGPU
using Milepost7
using MPI
using Logging


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

if !isinteractive()
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
else
    # case = "cases/case_ACTIVSg10k.m"
    case = "cases/case9.m"
    load = "cases/case9"
    profile = true
    output = true
    T = 2
    K = 1
    rhopq = 3e3
    rhova = 3e4
    configuration = 4
    exatron_inner_iter = 400
    exatron_outer_iter = 1
    proxal_iter = 2
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
