using ComponentArrays
using Plots
using StatsPlots
using Distributions
using Random

#save figures after running?
saving = true
save_path = "./UpdatedModifiers"

#files to read data from, relative to 
files = [["./UpdatedModifiers/UpdatedModifiers_lesslogscale_PKVPA_$(i)_true.txt" for i in 1:3], ["./UpdatedModifiers/UpdatedModifiers_PKVPA_$(i)_true.txt" for i in 1:3]]
files2 = [["./UpdatedModifiers/UpdatedModifiers_lesslogscale_PKVPA_$(i)_false.txt" for i in 1:3], ["./UpdatedModifiers/UpdatedModifiers_PKVPA_$(i)_false.txt" for i in 1:3]]
#Do space before name if not empty, else empty string
names_distinction = (" no updates", " updates")
#models each subarray corresponds to
models = ["VPA less logscale", "VPA"]
#colour for each model
colours = [[:magenta, :red], [:purple, :salmon]]
#quantity of interest
quant = "modifiers with updates logscale"
short_quant = "UpdatedModifierslogscale"
#corresponding values of interest for each model
values = [[1,2,3] for model in models]#Legend setting for plotting
legendcolumns = isempty(names_distinction[1]) || isempty(names_distinction[2]) ? 2 : 1
#set variable of interest below

upper_plotting_bound = [Inf for model in models]
upper_outlier_bound = [Inf for model in models]
spaced_accordingly = false
plot_separate = false

confidence = 0.95
q = quantile(Normal(), (1-(1-confidence)/2))
default(guidefontsize = 12, legendfontsize = 11)
verbose = false
CI_for_means = true
both_CIs = false
CI_plotting = true
CI_cutoff = Inf #3000.0

#Note for more than two in to read, only first two will be plotted against each other
to_read = (files, files2)
estimates_all = [[[] for model in to_read[k]] for k in eachindex(to_read)]
abs_errors_all = [[[] for model in to_read[k]] for k in eachindex(to_read)]
rel_errors_all = [[[] for model in to_read[k]] for k in eachindex(to_read)]
mean_squared_errors_all = [[[] for model in to_read[k]] for k in eachindex(to_read)]
rel_squared_errors_all = [[[] for model in to_read[k]] for k in eachindex(to_read)]
times_all = [[[] for model in to_read[k]] for k in eachindex(to_read)]
obj_diffs_all = [[[] for model in to_read[k]] for k in eachindex(to_read)]
CIs_all = [[[] for model in to_read[k]] for k in eachindex(to_read)]
for k in eachindex(to_read)
    for j in eachindex(to_read[k]) 
        for file in to_read[k][j]
            estimates = []
            abs_errors = []
            rel_errors = []
            mean_squared_errors = []
            rel_squared_errors = []
            times = []
            obj_diffs = []
            CIs = []
            lines = readlines(file)
            for i in eachindex(lines)
                if occursin("Estimates:",lines[i])
                    push!(estimates, eval(Meta.parse(lines[i+1])))
                elseif occursin("abs_errors:",lines[i])
                    #Meta.parse cannot handle ComponentArray, so need to get rid of this here
                    text = lines[i+1]
                    text = replace(text, "ComponentArray" => "")
                    append!(abs_errors, eval(Meta.parse(text)))
                elseif occursin("rel_errors:",lines[i])
                    #Meta.parse cannot handle ComponentArray, so need to get rid of this here
                    text = lines[i+1]
                    text = replace(text, "ComponentArray" => "")
                    append!(rel_errors, eval(Meta.parse(text)))
                elseif occursin("mean_squared_errors:", lines[i])
                    append!(mean_squared_errors, eval(Meta.parse(lines[i+1])))
                elseif occursin("rel_squared_errors:",lines[i])
                    append!(rel_squared_errors, eval(Meta.parse(lines[i+1])))
                elseif occursin("times:",lines[i])
                    append!(times, eval(Meta.parse(lines[i+1])))
                elseif occursin("obj_diffs:",lines[i])
                    append!(obj_diffs, eval(Meta.parse(lines[i+1])))
                elseif occursin("CIs:",lines[i])
                    #Here need to get rid of long type declaration in beginning for parser
                    text = lines[i+1]
                    index = findlast("}", text)
                    text = text[(collect(index)[1]+1):end]
                    append!(CIs, eval(Meta.parse(text)))
                end
            end
            estimates = [ComponentArray(PK = ComponentArray(est.PK), Seizure = ComponentArray(est.Seizure)) for est in estimates]
            push!(estimates_all[k][j], estimates)
            push!(abs_errors_all[k][j], abs_errors)
            push!(rel_errors_all[k][j], rel_errors)
            push!(mean_squared_errors_all[k][j], mean_squared_errors)
            push!(rel_squared_errors_all[k][j], rel_squared_errors)
            push!(times_all[k][j], times)
            push!(obj_diffs_all[k][j], obj_diffs)
            push!(CIs_all[k][j], CIs)
        end
    end
end

#pick variable of interest
interests = [[[[(j == 1 ? est.PK.c3 : est.PK.c2) for est in estes] for estes in rel_errors_all[k][j]] for j in eachindex(to_read[k])] for k in eachindex(to_read)]
#give name
name = "dose parameter relative error"
shorter_name = "relative error"
short_name = "dose_param"

function in_interval(x, y)
    if x isa Number 
        return (y[1] ≤ x ≤ y[2])
    else
        ins = Vector{Bool}(undef, length(x))
        for i in eachindex(x)
            ins[i] = (y[i][1] ≤ x[i] ≤ y[i][2])
        end
        return ins
    end
end

function size_Tuple(x)
    if x isa Tuple || eltype(x) <: Number
        return abs(x[2]-x[1])
    else
        a = Vector{Float64}(undef, length(x))
        for i in eachindex(x)
            a[i] = abs(x[i][2]-x[i][1])
        end
        return max(a...)
    end
end

function size_CI(x)
    return max(vcat([size_Tuple(x.PK[key]) for key in keys(x.PK)],[size_Tuple(x.Seizure[key]) for key in keys(x.Seizure)])...)
end

function finite_Tuple(y)
    if y isa Number
        return isfinite(y)
    else
        return all(isfinite.(y))
    end
end

function finite_CI(x)
    PKs = [finite_Tuple(x.PK[key]) for key in keys(x.PK)]
    Seizures = [x.Seizure[key] isa Number ? isfinite(x.Seizure[key]) : all([finite_Tuple(y) for y in x.Seizure[key]]) for key in keys(x.Seizure)]
    return all(PKs)&&all(Seizures)
end

function perm_test(A, B, tries::Int = 1000)
    m_A = mean(A)
    m_B = mean(B)
    diff = m_A - m_B
    full = [deepcopy(A)..., deepcopy(B)...]
    assigner = [zeros(length(A))..., ones(length(B))...]
    new_diffs = Vector{Float64}(undef, tries)
    #ensures p>0
    new_diffs[1] = diff
    for t in 2:tries
        perm = randperm(length(assigner))
        new_A = [full[j] for j in eachindex(full) if assigner[perm[j]] == 0]
        new_B = [full[j] for j in eachindex(full) if assigner[j] == 1]
        new_diffs[t] = mean(new_A) - mean(new_B)
    end
    p_different_mean = length([new_diffs[t] for t in 1:tries if abs(new_diffs[t])>=abs(diff)])/tries
    return p_different_mean
end

#Calculate coverage
coverage_full = [[[] for model in models], [[] for model in models]]
for k in eachindex(models)
    for n in eachindex(to_read)
        if isassigned(estimates_all[n],k) && !isempty(estimates_all[n][k])
            for point in eachindex(estimates_all[n][k])
                coverages = []
                for inst in eachindex(estimates_all[n][k][point])
                    axes = getaxes(estimates_all[n][k][point][inst])
                    coverage = []
                    for key in keys(CIs_all[n][k][point][inst])
                        covered = ComponentArray(PK = ComponentArray([in_interval(estimates_all[n][k][point][inst].PK[key2], CIs_all[n][k][point][inst][key].PK[key2]) for key2 in keys(CIs_all[n][k][point][inst][key].PK)], getaxes(estimates_all[n][k][point][inst].PK)), 
                                    Seizure = ComponentArray([a for key2 in keys(CIs_all[n][k][point][inst][key].Seizure) for a in in_interval(estimates_all[n][k][point][inst].Seizure[key2],CIs_all[n][k][point][inst][key].Seizure[key2])], getaxes(estimates_all[n][k][point][inst].Seizure)))
                        push!(coverage, covered)
                    end
                    coverage = NamedTuple{keys(CIs_all[n][k][point][inst])}(coverage)
                    push!(coverages, coverage)
                end
                push!(coverage_full[n][k], coverages)
            end
        end
    end
end

println("Coverage overall: ")
for k in eachindex(models)
    for n in eachindex(to_read)
        if isassigned(estimates_all[n],k) && !isempty(estimates_all[n][k])
            for point in eachindex(estimates_all[n][k])
                for key in keys(coverage_full[n][k][point][1])
                    coverages = [all([entry for entry in inst[key]]) for inst in coverage_full[n][k][point]]
                    println("Coverage for model $(models[k]) and point $(values[k][point])$(names_distinction[n]), $(key): ", sum(coverages)/length(coverages))
                end
            end
        end
    end
end
println()

#Determine infinite CIs
println("Infinite CIs:")
for k in eachindex(models)
    for n in eachindex(to_read)
        if isassigned(estimates_all[n],k) && !isempty(estimates_all[n][k])
            for point in eachindex(estimates_all[n][k])
                for key in keys(CIs_all[n][k][point][1])
                    finites = [!finite_CI(entry[key]) for entry in CIs_all[n][k][point]]
                    println("Infinite CIs for model $(models[k]) and point $(values[k][point])$(names_distinction[n]), $(key): ", sum(finites)/length(finites))
                end
            end
        end
    end
end
println()

#Determine sizes CIs
sizes_full = [[[[] for value in values[model]] for model in eachindex(models)], [[[] for value in values[model]] for model in eachindex(models)]]
mean_CI_size_finite = [[[[] for value in values[model]] for model in eachindex(models)], [[[] for value in values[model]] for model in eachindex(models)]]
println("Sizes CIs:")
for k in eachindex(models)
    for n in eachindex(to_read)
        if isassigned(estimates_all[n],k) && !isempty(estimates_all[n][k])
            for point in eachindex(estimates_all[n][k])
                for key in keys(CIs_all[n][k][point][1])
                    sizes = [size_CI(entry[key]) for entry in CIs_all[n][k][point]]
                    push!(sizes_full[n][k][point], sizes)
                    println("Average size CIs for model $(models[k]) and point $(values[k][point])$(names_distinction[n]), $(key): ", sum(sizes)/length(sizes))
                    sizes2 = [size for size in sizes if isfinite(size)]
                    println("Average size CIs for model $(models[k]) and point $(values[k][point])$(names_distinction[n]), $(key) for finite: ", sum(sizes2)/length(sizes2))
                    println("Maximal size CIs for model $(models[k]) and point $(values[k][point])$(names_distinction[n]), $(key) for finite: ", max((isempty(sizes2) ? NaN : sizes2)...))
                    println("Minimal size CIs for model $(models[k]) and point $(values[k][point])$(names_distinction[n]), $(key) for finite: ", min((isempty(sizes2) ? NaN : sizes2)...))
                    push!(mean_CI_size_finite[n][k][point], sum(sizes2)/length(sizes2))
                end
            end
        end
    end
end
println()

if CI_plotting
    pl3 = plot(xlabel = uppercasefirst(quant), ylabel = "means of finite CI size", title = (verbose ? "Means of finite CI size for \n different "*lowercasefirst(quant) : "Means of finite CI size"), ylims = (0.0, CI_cutoff))
    for k in eachindex(models)
        for n in 1:2
            if !isempty(CIs_all[n])
                if !isempty(CIs_all[n][k]) && !isempty(CIs_all[n][k][1])
                    for key in eachindex(keys(CIs_all[n][k][1][1]))
                        mean_sizes = [point[key] for point in mean_CI_size_finite[n][k]]
                        full_sizes = [[inst[key] for inst in point] for point in sizes_full[n][k]]
                        #err_lower = q*[std(full_sizes[i])/sqrt(length(full_sizes[i])) for i in eachindex(sizes_full[n][k])]
                        #err_upper = err_lower
                        #plot!(values[k], means_full[n][k], linecolor = colours[n][k], linewidth = 2, label = "means for "*models[k]*names_distinction[n], yerror=(err_lower, err_upper))
                        if any(isfinite.(mean_sizes)) && any([in_interval(mean, (0.0, CI_cutoff)) for mean in mean_sizes])
                            scatter!(values[k], mean_sizes, linecolor = colours[n][k], alpha = 1/key, mc = 2, label = "")
                            plot!(values[k], mean_sizes, linecolor = colours[n][k], alpha = 1/key, linewidth = 2, label = "means for "*models[k]*names_distinction[n]*", $(keys(CIs_all[n][k][1][1])[key])")
                        end
                    end
                end
            end
        end
        if eltype(values[k]) <: Number
            plot!(xticks = values[k], xrotation = 75)
        end
        plot!(legend=:outerbottom, legendcolumns=1)
    end
    display(pl3)
end

#=
#Do space before name if not empty, else empty string
frequent_or_sampled = 2
if frequent_or_sampled == 1
    names_distinction = (" inverse hessian", " sandwich")
else
    names_distinction = (" quantiles", " hpd")
end
#pick variable of interest
interests = [[[[size[n] for size in sizes_full[frequent_or_sampled][k][1]]] for k in eachindex(models)] for n in 1:2]
#give name
name = "size of credible intervals"
shorter_name = "CI size"
short_name = "CI"
=#

if spaced_accordingly
    values2 = deepcopy(values)
else
    values2 = [String.(Symbol.(value)) for value in values]
end

#per model plots
per_model_plots = [Plots.Plot[] for model in models]
#store means somewhere
means_full = [[[] for model in models], [[] for model in models]]
CI_means_full = [[[] for model in models], [[] for model in models]]
means_truncated = [[[] for model in models],[[] for model in models]]
for k in eachindex(models)
    if k ≤ length(interests[1]) && !isempty(interests[1][k])
        interest = interests[1][k]
    else
        interest = nothing
    end
    if k ≤ length(interests[2]) && !isempty(interests[2][k])
        interest_second = interests[2][k]
    else
        interest_second = nothing
    end
    interest_list = (interest, interest_second)
    label_set = [false,false]
    pl = plot(xlabel = uppercasefirst(quant), ylabel = (verbose ? uppercasefirst(name) : uppercasefirst(shorter_name)), title = (verbose ? (uppercasefirst(name)*" for different \n "*lowercasefirst(quant)*" in "*models[k]) : ""))
    interest_for_perm = Vector{Any}(undef, 2)
    for n in eachindex(interest_list)
        if !isnothing(interest_list[n])
            #Mention how handle outliers
            interest2 = deepcopy(interest_list[n])
            for i in eachindex(interest2)
                interest2[i] = [rel for rel in interest2[i] if rel <= upper_plotting_bound[k]]
            end
            interest3 = deepcopy(interest_list[n])
            for i in eachindex(interest3)
                interest3[i] = [rel for rel in interest3[i] if rel <= upper_outlier_bound[k]]
                #print outliers
                println("Outliers for $(values[k][i]) in model "*models[k]*" $(names_distinction[n]) by outlier bound: ", [rel for rel in interest_list[n][i] if rel > upper_outlier_bound[k]])
                println("Outliers for $(values[k][i]) in model "*models[k]*" $(names_distinction[n]) by plotting bound: ", [rel for rel in interest_list[n][i] if rel > upper_plotting_bound[k]])
            end
            interest_for_perm[n] = interest3
            for i in eachindex(interest2)
                if n==1 && !isnothing(interest_list[2]) && (i ≤ length(interest_list[2])) && !isempty(interest_list[2][i]) 
                    label = (label_set[n]) ? "" : names_distinction[n]
                    if !isempty(interest2[i])
                        violin!([values2[k][i]], interest2[i], side = :left, outliers=false, label = label, alpha = 0.5, color = colours[n][k])
                        label_set[n] = true
                    end
                elseif n==2 && !isnothing(interest_list[1]) && (i ≤ length(interest_list[1])) && !isempty(interest_list[1][i]) 
                    label = (label_set[n]) ? "" : names_distinction[n]
                    if !isempty(interest2[i])
                        violin!([values2[k][i]], interest2[i], side = :right, outliers=false, label = label, alpha = 0.5, color = colours[n][k])
                        label_set[n] = true
                    end
                else
                    if !isempty(interest2[i])
                        violin!([values2[k][i]], interest2[i], outliers=false, label = "", alpha = 0.5, color = colours[n][k])
                    end
                end
            end
            #Calculate means and store for later
            means = [(i ≤ length(interest3) && !isempty(interest3[i])) ? mean(interest3[i]) : NaN for i in eachindex(values2[k])]
            CI_means = [(i ≤ length(interest3) && !isempty(interest3[i])) ? (mean(interest3[i]) - q*std(interest3[i])/sqrt(length(interest3[i])), mean(interest3[i]) + q*std(interest3[i])/sqrt(length(interest3[i]))) : (-Inf, Inf) for i in eachindex(values2[k])]
            means2 = [(i ≤ length(interest2) && !isempty(interest2[i])) ? mean(interest2[i]) : NaN for i in eachindex(values2[k])]
            push!(means_truncated[n][k], means2)
            push!(means_full[n][k], means)
            push!(CI_means_full[n][k], CI_means)
            err_lower = means_full[n][k][1] .- [CI[1] for CI in CI_means_full[n][k][1]]
            err_upper = [CI[2] for CI in CI_means_full[n][k][1]] .- means_full[n][k][1]

            #plot!(values2, means2, linecolor = colours[k], linewidth = 2, label = "means of all lower than $(upper_plotting_bound[k]) for "*models[k])
            if (!CI_for_means || both_CIs)
                scatter!(values2[k], means, markercolor = colours[n][k], markershape = :star5, linewidth = 2, label = "means overall for "*models[k]*names_distinction[n], yerror = (err_lower, err_upper))
            else
                scatter!(values2[k], means, markercolor = colours[n][k], markershape = :star5, linewidth = 2, label = "means overall for "*models[k]*names_distinction[n])
            end
            if isfinite(upper_plotting_bound[k])
                scatter!(values2[k], means2, markercolor = colours[n][k], markershape = :circle, linewidth = 1, alpha = 0.7, label = "means of all lower than $(upper_plotting_bound[k]) for "*models[k]*names_distinction[n])
            end
            #Plot untruncated means only in full one?
            #plot!(values2, means2, linecolor = :blue, linewidth = 2, label = "means")
            if spaced_accordingly
                plot!(xticks = values2[k], xrotation = 90)
            end
        end
    end
    #println(interest_for_perm, isassigned(interest_for_perm))
    if all([isassigned(interest_for_perm,i) for i in eachindex(interest_for_perm)])
        shorter = (length(interest_for_perm[1]) < length(interest_for_perm[2]) ? interest_for_perm[1] : interest_for_perm[2])
        for i in eachindex(shorter)
            if !isempty(interest_for_perm[1][i]) && !isempty(interest_for_perm[2][i])
                p = perm_test(interest_for_perm[1][i], interest_for_perm[2][i])
                println("P-value for mean difference for $(values[k][i]) in model "*models[k]*": ", p)
                end
        end
    end
    plot!(legend=:outerbottom, legendcolumns=1)
    push!(per_model_plots[k],pl)
    display(pl)
end

#Overall means plots
overall_plots = Plots.Plot[]
pl2 = plot(xlabel = quant, ylabel = "means of "*(verbose ? lowercasefirst(name) : lowercasefirst(shorter_name)), title = (verbose ? ("Truncated means of "*lowercasefirst(name)*" for \n different "*lowercasefirst(quant)) : "Truncated means"))
for k in eachindex(models)
    for n in 1:2
        if !isempty(means_truncated[n][k])
            plot!(values[k], means_truncated[n][k], linecolor = colours[n][k], linewidth = 2, label = "means of all lower than $(upper_plotting_bound[k]) for "*models[k]*names_distinction[n])
        end
    end
    if eltype(values[k]) <: Number
        plot!(xticks = values[k], xrotation = 75)
    end
    plot!(legend=:outerbottom, legendcolumns=legendcolumns)
end
push!(overall_plots, pl2)
display(pl2)

if plot_separate && any(.!isempty.(means_truncated[1])) && any(.!isempty.(means_truncated[2]))
    for n in 1:2
        pl25 = plot(xlabel = quant, ylabel = "means of "*(verbose ? lowercasefirst(name) : lowercasefirst(shorter_name)), title = (verbose ? ("Truncated means of "*lowercasefirst(name)*" for \n different "*lowercasefirst(quant)*names_distinction[n]) : "Truncated means"))
        for k in eachindex(models)
            if !isempty(means_truncated[n][k])
                plot!(values[k], means_truncated[n][k], linecolor = colours[n][k], linewidth = 2, label = "means of all lower than $(upper_plotting_bound[k]) for "*models[k]*names_distinction[n])
            end
            if eltype(values[k]) <: Number
                plot!(xticks = values[k], xrotation = 75)
            end
            plot!(legend=:outerbottom, legendcolumns=legendcolumns)
        end
        push!(overall_plots, pl25)
        display(pl25)
    end
end

pl3 = plot(xlabel = uppercasefirst(quant), ylabel = "means of "*(verbose ? lowercasefirst(name) : lowercasefirst(shorter_name)), title = (verbose ? ("Overall means of "*lowercasefirst(name)*" for \n different "*lowercasefirst(quant)) : "Overall means"))
for k in eachindex(models)
    for n in 1:2
        if !isempty(means_truncated[n][k])
            err_lower = means_full[n][k][1] .- [CI[1] for CI in CI_means_full[n][k][1]]
            err_upper = [CI[2] for CI in CI_means_full[n][k][1]] .- means_full[n][k][1]
            if (CI_for_means || both_CIs)
                plot!(values[k], means_full[n][k], linecolor = colours[n][k], linewidth = 2, label = "means of all for "*models[k]*names_distinction[n], yerror=(err_lower, err_upper))
            else
                plot!(values[k], means_full[n][k], linecolor = colours[n][k], linewidth = 2, label = "means of all for "*models[k]*names_distinction[n])
            end
        end
    end
    if eltype(values[k]) <: Number
        plot!(xticks = values[k], xrotation = 75)
    end
    plot!(legend=:outerbottom, legendcolumns=legendcolumns)
end
push!(overall_plots, pl3)
display(pl3)

if plot_separate && any(.!isempty.(means_full[1])) && any(.!isempty.(means_full[2]))
    for n in 1:2
        pl35 = plot(xlabel = quant, ylabel = "means of "*(verbose ? lowercasefirst(name) : lowercasefirst(shorter_name)), title = (verbose ? ("Overall means of "*lowercasefirst(name)*" for \n different "*lowercasefirst(quant)*names_distinction[n]) : "Overall means"))
        for k in eachindex(models)
            if !isempty(means_full[n][k])
                #plot!(values[k], means_full[n][k], linecolor = colours[n][k], linewidth = 2, label = "means of all for "*models[k]*names_distinction[n])
                err_lower = means_full[n][k][1] .- [CI[1] for CI in CI_means_full[n][k][1]]
                err_upper = [CI[2] for CI in CI_means_full[n][k][1]] .- means_full[n][k][1]
                if (CI_for_means || both_CIs)
                    plot!(values[k], means_full[n][k], linecolor = colours[n][k], linewidth = 2, label = "means of all for "*models[k]*names_distinction[n], yerror=(err_lower, err_upper))
                else
                    plot!(values[k], means_full[n][k], linecolor = colours[n][k], linewidth = 2, label = "means of all for "*models[k]*names_distinction[n])
                end
            end
            if eltype(values[k]) <: Number
                plot!(xticks = values[k], xrotation = 75)
            end
            plot!(legend=:outerbottom, legendcolumns=legendcolumns)
        end
        push!(overall_plots, pl35)
        display(pl35)
    end
end

if saving
    for i in eachindex(per_model_plots)
        savefig(per_model_plots[i][1], joinpath(save_path,"$(models[i])_$(short_quant)_$(short_name).png"))
    end
    savefig(overall_plots[1], joinpath(save_path,"Truncated_means_$(short_quant)_$(short_name).png"))
    i=1
    if plot_separate && any(.!isempty.(means_truncated[1])) && any(.!isempty.(means_truncated[2]))
        savefig(overall_plots[i+1], joinpath(save_path,"Truncated_means_$(short_quant)_$(short_name)$(names_distinction[1]).png"))
        savefig(overall_plots[i+2], joinpath(save_path,"Truncated_means_$(short_quant)_$(short_name)$(names_distinction[2]).png"))
        i+=2
    end
    savefig(overall_plots[i+1], joinpath(save_path,"Overall_means_$(short_quant)_$(short_name).png"))
    i+=1
    if plot_separate && any(.!isempty.(means_full[1])) && any(.!isempty.(means_full[2]))
        savefig(overall_plots[i+1], joinpath(save_path,"Overall_means_$(short_quant)_$(short_name)$(names_distinction[1]).png"))
        savefig(overall_plots[i+2], joinpath(save_path,"Overall_means_$(short_quant)_$(short_name)$(names_distinction[2]).png"))
        i+=2
    end
end

#=
#Do plots for wrong model
model_colours = [:violet, :teal, :orange]
model_colours_other = [:teal, :violet, :teal]
model_names = ["NegativeBinomial", "Basic", "VPA"]
model_names_other = ["Basic", "NegativeBinomial", "Basic"]
plots = [Plots.Plot[] for model in models]
comp_plots = Plots.Plot[]
estimates = [ComponentArray(PK = ComponentArray(estimates_all[1][k][1][1].PK), Seizure = ComponentArray(estimates_all[1][k][1][1].Seizure)) for k in eachindex(files)]
for k in eachindex(files)
    for j in eachindex(estimates_all[1][k][1][1].Seizure)
        pl = plot(title = "Distribution of $(j)", ylabel = "Parameter", xtickfontsize = 11)
        param_estimates = [(j == :b ? estimate.Seizure[j][1] : estimate.Seizure[j]) for estimate in estimates_all[1][k][1]]
        if !(j== :b)
            estimates[k].Seizure[j] = mean(param_estimates)
        else
            estimates[k].Seizure[j] = [mean(param_estimates)]
        end
        violin!([String(j)], param_estimates, outliers=false, label = "", alpha = 0.5, color = model_colours[k])
        display(pl)
        push!(plots[k], pl)
        if saving
            savefig(pl, joinpath(save_path,"$(model_names[k])_on_$(model_names_other[k])_param_$(j).png"))
        end
    end
end

#Comparison plots
using Pkg
include("EpilepsyModels.jl")

using .EpilepsyModels
using ComponentArrays
using OptimizationOptimJL
using OptimizationBBO
using LineSearches
using DifferentialEquations
using Plots
using StatsPlots
using StaticArrays
using Random
using Distributions
using BenchmarkTools
using ModelingToolkit
using MCMCChains
using AdvancedHMC

Input_θ_PKBasic = ComponentArray((k_el = 2.0, k_abs = 5.0, σ=0.2))
Input_θ_PKLEV = ComponentArray((k_abs = (24*3.5), c1 = (24*4.0), c2 = 0.25, c3 = 0.122, v1 = 29.7, v2 = 2.85, σ=0.2))
Input_θ_PKLEVNoAbsorption = ComponentArray((c1 = (24*4.0), c2 = 0.25, c3 = 0.122, v1 = 29.7, v2 = 2.85, σ=0.2))
Input_θ_PKCBZ = ComponentArray((k_abs = (24*0.45), c1 = (24*1.96), c2 = 1.73, c3 = 24*1.36, v1 = 164.0/75.0, σ=0.2))
Input_θ_PKVPA = ComponentArray((k_abs = (24*1.86), c1 = (24*0.577), c2 = 0.535, c3 = 0.875, v1 = 0.28, σ=0.2))
Input_θ_PKLTG = ComponentArray((k_abs = (24*1.96), c1 = (24*2.4), c2 = 0.938, c3 = 110*0.00328, c4 = 0.34, v1 = 2.14, σ=0.2))
Input_θ_PKBigFour = ComponentArray((k_abs_LTG = (24*1.96), c1_LTG = (24*2.4), c2_LTG = 0.938, c3_LTG = 110*0.00328, c4_LTG = 0.34, c_Inh_LTG = (1-0.579), c_Ind_LTG = (1+0.546), v1_LTG = 2.14, σ_LTG=0.2,
            k_abs_VPA = (24*1.86), c1_VPA = (24*0.577), c2_VPA = 0.535, c3_VPA = 0.875, c_Ind_VPA = 1.22, v1_VPA = 0.28, σ_VPA=0.2,
            k_abs_CBZ = (24*0.45), c1_CBZ = (24*1.96), c2_CBZ = 1.73, c3_CBZ = 24*1.36, v1_CBZ = 164.0/75.0, σ_CBZ=0.2,
            k_abs_LEV = (24*3.5), c1_LEV = (24*4.0), c2_LEV = 0.25, c3_LEV = 0.122, c_Inh_LEV = 0.812, c_Ind_LEV = 1.09, v1_LEV = 29.7, v2_LEV = 2.85, σ_LEV=0.2))
#Seizure Models
base_rate = 4.0
Input_θ_SeizureBasic_one = ComponentArray((a = base_rate, b = SA[0.2]))
Input_θ_SeizureBasic_four = ComponentArray((a = base_rate, b = SA[base_rate/7, base_rate/25, base_rate/20, base_rate/120]))
Input_θ_SeizureNegativeBinomial = ComponentArray((a = log(4.0), o = 1.128, prev = 0.731, b = SA[0.2]))
Input_θ_SeizureVPA = ComponentArray((a = 6.1, a1 = 1.0, a2 = 1.8, b1 = 13.3, b2 = 2.4))
Input_θ_SeizureSANAD_one = ComponentArray((a1 = log(1.09), a2 = log(0.87), a3 = log(1.15), b = SA[7/30])) 
Input_θ_SeizureSANAD_four = ComponentArray((a1 = log(1.09), a2 = log(0.87), a3 = log(1.15), b = SA[6.75/9, 7.5/29, 7.25/8, 7.25/75]))

Population_size = 1
wo_treatment = 0.0 #10.0
Obs_Duration = wo_treatment + 20.0
PK_timepoints = wo_treatment:3.75:Obs_Duration
drug_appropriate_dosing = true
max_threads_simul = Threads.nthreads()
ODE_options = (AutoTsit5(Rosenbrock23()),)
time = [(0.0, 8.0), (0.0, 8.0), (0.0, 20.0)]

for k in eachindex(files)
    Random.seed!(42)
    parsed2 = k

    if parsed2 in [1,2]
        Seizure_timepoints = 0.0:1.0:Obs_Duration
    else
        Seizure_timepoints = 0.0:5.0:Obs_Duration
    end
    no_counts_seizure = false
    update_reg = Obs_Duration
    if parsed2 in [1,2]
        pk_model = PKLEV(θ=Input_θ_PKLEV)
    end
    if parsed2 == 3
        pk_model = PKVPA(θ=Input_θ_PKVPA)
    end
    if parsed2 in [1,3]
        if (typeof(pk_model).name.wrapper in [PKVPA])
            Input_θ_SeizureBasic_one.b = SA[0.05]
        end
        if (typeof(pk_model).name.wrapper in [PKBigFour])
            seizure_model = SeizureBasic(θ = Input_θ_SeizureBasic_four)
        else
            seizure_model = SeizureBasic(θ = Input_θ_SeizureBasic_one)
            if drug_appropriate_dosing
                if (typeof(pk_model).name.wrapper in [PKLEV, PKLEVNoAbsorption])
                    Input_θ_SeizureBasic_one.b = SA[Input_θ_SeizureBasic_four.b[2]]
                elseif (typeof(pk_model).name.wrapper in [PKLTG])
                    Input_θ_SeizureBasic_one.b = SA[Input_θ_SeizureBasic_four.b[1]]
                elseif (typeof(pk_model).name.wrapper in [PKCBZ])
                    Input_θ_SeizureBasic_one.b = SA[Input_θ_SeizureBasic_four.b[3]]
                elseif (typeof(pk_model).name.wrapper in [PKVPA])
                    Input_θ_SeizureBasic_one.b = SA[Input_θ_SeizureBasic_four.b[4]]
                end
            end
        end
    elseif parsed2 == 2
        if (typeof(pk_model).name.wrapper in [PKVPA])
            Input_θ_SeizureBasic_one.b = SA[0.05]
        end
        if (typeof(pk_model).name.wrapper in [PKBigFour])
            seizure_model2 = SeizureBasic(θ = Input_θ_SeizureBasic_four)
        else
            seizure_model2 = SeizureBasic(θ = Input_θ_SeizureBasic_one)
            if drug_appropriate_dosing
                if (typeof(pk_model).name.wrapper in [PKLEV, PKLEVNoAbsorption])
                    Input_θ_SeizureBasic_one.b = SA[Input_θ_SeizureBasic_four.b[2]]
                elseif (typeof(pk_model).name.wrapper in [PKLTG])
                    Input_θ_SeizureBasic_one.b = SA[Input_θ_SeizureBasic_four.b[1]]
                elseif (typeof(pk_model).name.wrapper in [PKCBZ])
                    Input_θ_SeizureBasic_one.b = SA[Input_θ_SeizureBasic_four.b[3]]
                elseif (typeof(pk_model).name.wrapper in [PKVPA])
                    Input_θ_SeizureBasic_one.b = SA[Input_θ_SeizureBasic_four.b[4]]
                end
            end
        end
    end
    if parsed2 == 3
        seizure_model2 = SeizureVPA(θ = Input_θ_SeizureVPA)
    end
    if pk_model isa PKVPA
        Input_θ_SeizureNegativeBinomial.b[1] = 0.05
    end
    if parsed2 == 2
        seizure_model = SeizureNegativeBinomial(θ = Input_θ_SeizureNegativeBinomial)
    elseif parsed2 == 1
        seizure_model2 = SeizureNegativeBinomial(θ = Input_θ_SeizureNegativeBinomial)
    end

    person_gen = BigFourPersonGenerator()
    dose_gen = PolyDosesRandom(pk_model, drug_appropriate_dosing)
    if (seizure_model isa SeizureVPA || seizure_model2 isa SeizureVPA) && !(dose_gen isa BigFourDoses)
        dose_distr = (d_VPA = (min = 150.0, avg_num = 8.0, max_num = 14), d_CBZ = (min = 200.0, avg_num = 3.0, max_num = 8))
        distr_first = (d_VPA = 1.0, d_CBZ = 0.0)
        distr_second = (d_VPA = 0.0, d_CBZ = 1.0)
        dose_gen = PolyDosesRandom(dose_distr, distr_first, distr_second; prob_second=0.5, times_per_day_first=2, times_per_day_second=2, assign_not_supported = true)
    elseif seizure_model isa SeizureVPA
        dose_gen = BigFourDoses(order_male = ((:d_VPA,:d_CBZ), (:d_VPA,:d_CBZ)), order_female = ((:d_VPA,:d_CBZ), (:d_VPA,:d_CBZ)), prob_second = 0.5, prob_reassignment = 0.0)
    end
    mod = FullModel(pk_model, seizure_model, person_gen, dose_gen)
    #reset PK estimate to true values
    estimates[k].PK = mod.pk_model.θ
    mod2 = FullModel(pk_model, seizure_model2, person_gen, dose_gen)

    if parsed2 in [1,2]
        data = generate_data(mod, Population_size, Obs_Duration, timepoints_PK = PK_timepoints, timepoints_seizure = Seizure_timepoints, wo_treatment = wo_treatment, max_threads = max_threads_simul, just_Bool = no_counts_seizure, ODE_options = ODE_options)
    elseif parsed2 == 3
        #Handle seizurevpa records success and seizurebasic records failures
        invert_seizures(x::NamedTuple) = (time = x.time, count = (x.count <= 10))
        #For each 5 day interval check if less than expected baseline of 10 seizures halved
        data = generate_data(mod, Population_size, Obs_Duration, timepoints_PK = PK_timepoints, timepoints_seizure = Seizure_timepoints, wo_treatment = wo_treatment, max_threads = max_threads_simul, just_Bool = no_counts_seizure, ODE_options = ODE_options)
        for person in data
            person.seizure_counts .= invert_seizures.(person.seizure_counts)
        end
    end

    println("Covariates: ", data[1].covariates)
    println("Dosing: ")
    for k in 1:5
        println(data[1].dosing[k])
    end
    #if parsed2==3
    #empty!(data[1].dosing)
    #EpilepsyModels.assign_dose!(BasicDoses(150.0,2), data[1], names = pk_model.keys, timeframe = Obs_Duration)
    #Now plot for both, sols use true parameters for both
    sols = [EpilepsyModels.solve_PK(mod.pk_model,mod.pk_model.θ, data[i], endpoint = Obs_Duration, options = ODE_options) for i in eachindex(data)]
    i = 1

    indices = [index for index in eachindex(data[i].seizure_counts) if data[i].seizure_counts[index].time[1] >= time[parsed2][1] && data[i].seizure_counts[index].time[2] <= time[parsed2][2]]
    intervals = [data[i].seizure_counts[index].time for index in indices]
    pl2 = plot(xlabel = "day", ylabel = "Seizure Probability", title = (verbose ? "Comparison for $(model_names[parsed2]) estimated on \n true model $(model_names_other[parsed2]) for example person" : "Comparison"))
    data2 = deepcopy(data[i])
    if parsed2 == 3
        to_Int(x::NamedTuple) = (time = x.time, count = Int(x.count))
        for i in eachindex(data2.seizure_counts)
            data2.seizure_counts[1] = to_Int(data2.seizure_counts[i])
        end
    end
    samples_true = [EpilepsyModels.draw_data_samples(mod.seizure_model, sols[i], person=data2, interval=interval,names=pk_model.keys, θ = mod.seizure_model.θ, sample_nr=1000) for interval in intervals]
    samples_estimate = [EpilepsyModels.draw_data_samples(mod2.seizure_model, sols[i], person=data[i], interval=interval,names=pk_model.keys, θ = estimates[k].Seizure, sample_nr=1000) for interval in intervals]
    #Plot the violins
    if !(eltype(samples_true[1]) <: Bool || eltype(samples_estimate[1]) <: Bool)
        for j in eachindex(intervals)
            violin!(["$(intervals[j])"], Float64.(samples_true[j]), side = :left, label = (j==1 ? "True model and parameters" : ""), colour = model_colours_other[parsed2])
            violin!(["$(intervals[j])"], Float64.(samples_estimate[j]), side = :right, label = (j==1 ? "Wrong model estimate" : ""), colour = model_colours[parsed2])
        end  
    elseif eltype(samples_true[1]) <: Bool && !(eltype(samples_estimate[1]) <: Bool)
        samples_true_means = [mean(samples) for samples in samples_true]
        stringed = ["$(interval)" for interval in intervals]
        for j in eachindex(intervals)
            violin!(["$(intervals[j])"], Float64.(samples_estimate[j]), label = (j==1 ? "Wrong model estimate" : ""), colour = model_colours[parsed2])
        end
        plot!(twinx(), samples_true_means, label = "Mean success probability of \n true model and parameters", colour = model_colours_other[parsed2], linewidth = 5, grid = false, ylabel = "Success Probability", ylim = [0.0, 1.5])
    elseif eltype(samples_estimate[1]) <: Bool && !(eltype(samples_true[1]) <: Bool)
        samples_estimate_means = [mean(samples) for samples in samples_estimate]
        stringed = ["$(interval)" for interval in intervals]
        for j in eachindex(intervals)
            violin!(["$(intervals[j])"], Float64.(samples_true[j]), label = (j==1 ? "True model and parameters" : ""), colour = model_colours_other[parsed2], ylims = [0.0, 30.0])
        end
        if parsed2 == 3
            plot!(stringed, [10 for string in stringed], label = "Mean success probability of \n wrong model estimate", linecolour = model_colours[parsed2], linewidth = 3)
            plot!(stringed, [10 for string in stringed], label = "50% reduction threshold \n compared to baseline", linecolour = :black, linewidth = 3, legend = :topleft, legendfontsize=9, legendcolumns=2)
        end
        plot!(twinx(), stringed, samples_estimate_means, label = "", colour = model_colours[parsed2], linewidth = 5,  grid = false, ylabel = "Success Probability", ylim = [0.0, 1.5])
    else
        samples_true_means = [mean(samples) for samples in samples_true]
        samples_estimate_means = [mean(samples) for samples in samples_estimate]
        stringed = ["$(interval)" for interval in intervals]
        plot!(stringed, samples_true_means, label = "Mean success probability of \n true model and parameters", ylabel = "Success probability")
        plot!(stringed, samples_estimate_means, label = "Mean success probability of \n wrong model estimate")
    end
    display(pl2)
    push!(comp_plots, pl2)   
    if saving
        savefig(pl2, joinpath(save_path,"Comparison_$(model_names[k])_on_$(model_names_other[k]).png"))
    end 
#end        
end
=#