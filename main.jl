using Milepost7

MPI.Init()

if !isinteractive()
    args = parse_cmd()
    profile = args["profile"]
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

    milepost7(case, load, T, K, configuration; profile=profile, rhopq=rhopq, rhova=rhova, proxal_iter=proxal_iter, exatron_inner_iter=exatron_inner_iter, exatron_outer_iter=exatron_outer_iter)
end