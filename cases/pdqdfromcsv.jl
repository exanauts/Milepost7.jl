using ArgParse
using Dates
using DelimitedFiles
using Plots
using PowerModels

"""
    main(casefile::String, ploadfile::String, qloadfile::String; toplot=false)

Read the load data from the CSV files and write the data in the format
expected by the ProxAL.jl package.
"""
function main(casefile::String, ploadfile::String, qloadfile::String; toplot=false)

# function main(case::String, pdcsv::String, qdcsv; toplot=false)
    # Read the data from the CSV file
    input_pd = DelimitedFiles.readdlm("$ploadfile", ',', Any, '\n')
    input_qd = DelimitedFiles.readdlm("$qloadfile", ',', Any, '\n')

    # Extract the data from the input array
    # The first column is the time in hours
    load_buses_ids = Int64.(input_pd[1,2:end])
    pd_loads = Float64.(input_pd[2:end,2:end])
    qd_loads = Float64.(input_qd[2:end,2:end])
    nperiods = size(pd_loads,1)

    pd_dict = Dict{Int64,Array{Float64,1}}()
    qd_dict = Dict{Int64,Array{Float64,1}}()

    for (i, bus) in enumerate(load_buses_ids)
        pd_dict[bus] = pd_loads[:,i]
        qd_dict[bus] = qd_loads[:,i]
    end


    # Create the data dictionary
    network_data = PowerModels.parse_file("$casefile")
    buses = network_data["bus"]
    nbus = length(buses)
    busids = sort(parse.(Int64, collect(keys(buses))))

    out_pd_loads = zeros(nbus, nperiods)
    out_qd_loads = zeros(nbus, nperiods)

    for (i, bus) in enumerate(busids)
        if haskey(pd_dict, bus)
            out_pd_loads[i,:] .= pd_dict[bus]
            out_qd_loads[i,:] .= qd_dict[bus]
        end
    end

    pname, ext = splitext(ploadfile)
    qname, ext = splitext(qloadfile)

    DelimitedFiles.writedlm("$pname.Pd", out_pd_loads, ' ')
    DelimitedFiles.writedlm("$qname.Qd", out_pd_loads, ' ')

    firstperiod = Time(input_qd[2,1], "yyyy-mm-dd HH:MM:SS")
    secondperiod = Time(input_qd[3,1], "yyyy-mm-dd HH:MM:SS")

    timestep = Minute((secondperiod - firstperiod))
    sum_pd = sum(out_pd_loads, dims=1)
    sum_qd = sum(out_qd_loads, dims=1)
    loadplot = plot(
        sum_pd[1,:],
        label="Pd",
        title="Load profile $case\n$(input_pd[2,1])-$(input_pd[end,1])\n Timestep: $timestep",
        titlefontsize=10,
        xlabel="Period",
        ylabel="Load (MW)",
    )
    plot!(loadplot, sum_qd[1,:], label="Qd")
    savefig(loadplot, "$case.png")
end

function parse_cmd()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--plot"
            help = "Create a plot of the load"
            action = :store_true
        "case"
            help = "Case file"
            arg_type = String
            required = true
        "Pd"
            help = "Pd load file"
            arg_type = String
            required = true
        "Qd"
            help = "Qd load file"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

if !isinteractive()
    args = parse_cmd()
    main(case, pdload, qdload; toplot=toplot)
end