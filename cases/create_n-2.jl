using DelimitedFiles

incase = ARGS[1]
input = "$incase.Ctgs"
outcase = "$(incase)_2"
output = "$(outcase).Ctgs"
ctgs = DelimitedFiles.readdlm(input)

newctgs = Vector{Vector{Int64}}()
for (i, ctg) in enumerate(ctgs)
    for j in 1:i-1
        push!(newctgs, [ctg, ctgs[j]])
    end
end

run(Cmd(`cp $(incase).m $(outcase).m`))
run(Cmd(`cp $(incase).Pd $(outcase).Pd`))
run(Cmd(`cp $(incase).Qd $(outcase).Qd`))

DelimitedFiles.writedlm(output, newctgs, " ")
