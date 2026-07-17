using ComponentArrays
using Plots
using StatsPlots
using Distributions

#files to read data from, relative to 
files = [["./PK_reg/CBZ_075.txt", "./PK_reg/CBZ_1.txt", "./PK_reg/CBZ_375.txt", "./PK_reg/CBZ_7.txt"], 
         ["./PK_reg/LEV_075.txt", "./PK_reg/LEV_1.txt", "./PK_reg/LEV_375.txt", "./PK_reg/LEV_7.txt"],
         ["./PK_reg/VPA_075.txt", "./PK_reg/VPA_1.txt", "./PK_reg/VPA_375.txt", "./PK_reg/VPA_7.txt"],
         ["./PK_reg/LTG_075.txt", "./PK_reg/LTG_1.txt", "./PK_reg/LTG_375.txt", "./PK_reg/LTG_7.txt"]]
#models each subarray corresponds to
models = ["CBZ", "LEV", "VPA", "LTG"]
#colour for each model
colours = [:blue, :green, :red, :purple]
#quantity of interest
quant = "regularities of PK measurements"
#corresponding values of interest
values = [0.75, 1, 3.75, 7]
#Legend setting for plotting
legendcolumns = 2
#set variable of interest below

upper_plotting_bound = [2.0 for model in models]
upper_outlier_bound = [100.0 for model in models]
spaced_accordingly = false

estimates_all = [[] for model in files]
abs_errors_all = [[] for model in files]
rel_errors_all = [[] for model in files]
mean_squared_errors_all = [[] for model in files]
rel_squared_errors_all = [[] for model in files]
times_all = [[] for model in files]
obj_diffs_all = [[] for model in files]
CIs_all = [[] for model in files]
for i in eachindex(files) 
    for file in files[i]
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
        push!(estimates_all[i], estimates)
        push!(abs_errors_all[i], abs_errors)
        push!(rel_errors_all[i], rel_errors)
        push!(mean_squared_errors_all[i], mean_squared_errors)
        push!(rel_squared_errors_all[i], rel_squared_errors)
        push!(times_all[i], times)
        push!(obj_diffs_all[i], obj_diffs)
        push!(CIs_all[i], CIs)
    end
end


#pick variable of interest
interests = rel_squared_errors_all
#give name
name = "relative squared errors"


if spaced_accordingly
    values2 = deepcopy(values)
else
    values2 = String.(Symbol.(values))
end

#per model plots
per_model_plots = [Plots.Plot[] for model in models]
#store means somewhere
means_full = [[] for model in models]
means_truncated = [[] for model in models]
for k in eachindex(interests)
    interest = interests[k]
    #Mention how handle outliers
    interest2 = deepcopy(interest)
    for i in eachindex(interest2)
        interest2[i] = [rel for rel in interest2[i] if rel <= upper_plotting_bound[k]]
    end
    interest3 = deepcopy(interest)
    for i in eachindex(interest3)
        interest3[i] = [rel for rel in interest3[i] if rel <= upper_outlier_bound[k]]
        #print outliers
        println("Outliers for $(values[i]) in model "*models[k]*" by outlier bound: ", [rel for rel in interest[i] if rel > upper_outlier_bound[k]])
        println("Outliers for $(values[i]) in model "*models[k]*" by plotting bound: ", [rel for rel in interest[i] if rel > upper_plotting_bound[k]])
    end
    pl = plot(xlabel = uppercasefirst(quant), ylabel = uppercasefirst(name), title = (uppercasefirst(name)*" for different \n "*lowercasefirst(quant)*" in "*models[k]))
    for i in eachindex(values)
        violin!([values2[i]], interest2[i], outliers=false, label = "", alpha = 0.5, color = colours[k])
    end
    #Calculate means and store for later
    means = [mean(rel) for rel in interest3]
    means2 = [mean(rel) for rel in interest2]
    push!(means_truncated[k], means2)
    push!(means_full[k], means)

    plot!(values2, means2, linecolor = colours[k], linewidth = 2, label = "means of all lower than $(upper_plotting_bound[k]) for "*models[k])
    #Plot untruncated means only in full one?
    #plot!(values2, means2, linecolor = :blue, linewidth = 2, label = "means")
    if spaced_accordingly
        plot!(xticks = values2, xrotation = 90)
    end
    plot!(legend=:outerbottom, legendcolumns=1)
    push!(per_model_plots[k],pl)
    display(pl)
end

#Overall means plots
overall_plots = Plots.Plot[]
pl2 = plot(xlabel = quant, ylabel = "means of "*lowercasefirst(name), title = ("Truncated means of "*lowercasefirst(name)*" for \n different "*lowercasefirst(quant)))
for k in eachindex(models)
    plot!(values, means_truncated[k], linecolor = colours[k], linewidth = 2, label = "means of all lower than $(upper_plotting_bound[k]) for "*models[k])
    plot!(xticks = values, xrotation = 75)
    plot!(legend=:outerbottom, legendcolumns=legendcolumns)
end
push!(overall_plots, pl2)
display(pl2)

pl3 = plot(xlabel = uppercasefirst(quant), ylabel = "means of "*lowercasefirst(name), title = ("Overall means of "*lowercasefirst(name)*" for \n different "*lowercasefirst(quant)))
for k in eachindex(models)
    plot!(values, means_full[k], linecolor = colours[k], linewidth = 2, label = "means of all for "*models[k])
    plot!(xticks = values, xrotation = 75)
    plot!(legend=:outerbottom, legendcolumns=legendcolumns)
end
push!(overall_plots, pl3)
display(pl3)