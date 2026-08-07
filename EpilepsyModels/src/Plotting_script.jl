using ComponentArrays
using Plots
using StatsPlots
using Distributions

#save figures after running?
saving = false
save_path = "./SeizureFreq"

#files to read data from, relative to 
files = [["./SeizureFreq/SeizureFreq_Basic_1_false.txt", "./SeizureFreq/SeizureFreq_Basic_2_false.txt", "./SeizureFreq/SeizureFreq_Basic_5_false.txt", "./SeizureFreq/SeizureFreq_Basic_10_false.txt"], 
        ["./SeizureFreq/SeizureFreq_NB_1_false.txt", "./SeizureFreq/SeizureFreq_NB_2_false.txt", "./SeizureFreq/SeizureFreq_NB_5_false.txt", "./SeizureFreq/SeizureFreq_NB_10_false.txt"], 
        ]
files2 = []
#Do space before name if not empty, else empty string
names_distinction = ("", " just Bool")
#models each subarray corresponds to
models = ["Basic", "Negative Binomial"]
#colour for each model
colours = [[:blue, :green, :red, :purple], [:lightblue, :lightgreen, :orange, :pink]]
#quantity of interest
quant = "seizure measurement frequency"
short_quant = "Seizure_Freq"
#corresponding values of interest
values = [1,2,5,10]
#Legend setting for plotting
legendcolumns = isempty(names_distinction[1]) || isempty(names_distinction[2]) ? 2 : 1
#set variable of interest below

upper_plotting_bound = [2.0, 40.0]
upper_outlier_bound = [150.0, 300.0]
spaced_accordingly = false
plot_separate = true

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
interests = rel_squared_errors_all
#give name
name = "relative squared errors"
short_name = "RSE"

if spaced_accordingly
    values2 = deepcopy(values)
else
    values2 = String.(Symbol.(values))
end

#per model plots
per_model_plots = [Plots.Plot[] for model in models]
#store means somewhere
means_full = [[[] for model in models], [[] for model in models]]
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
    pl = plot(xlabel = uppercasefirst(quant), ylabel = uppercasefirst(name), title = (uppercasefirst(name)*" for different \n "*lowercasefirst(quant)*" in "*models[k]))
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
                println("Outliers for $(values[i]) in model "*models[k]*" $(names_distinction[n]) by outlier bound: ", [rel for rel in interest[i] if rel > upper_outlier_bound[k]])
                println("Outliers for $(values[i]) in model "*models[k]*" $(names_distinction[n]) by plotting bound: ", [rel for rel in interest[i] if rel > upper_plotting_bound[k]])
            end
            for i in eachindex(interest2)
                if n==1 && !isnothing(interest_list[2]) && (i ≤ length(interest_list[2])) && !isempty(interest_list[2][i]) 
                    label = (label_set[n]) ? "" : names_distinction[n]
                    violin!([values2[i]], interest2[i], side = :left, outliers=false, label = label, alpha = 0.5, color = colours[n][k])
                    label_set[n] = true
                elseif n==2 && !isnothing(interest_list[1]) && (i ≤ length(interest_list[1])) && !isempty(interest_list[1][i]) 
                    label = (label_set[n]) ? "" : names_distinction[n]
                    violin!([values2[i]], interest2[i], side = :right, outliers=false, label = label, alpha = 0.5, color = colours[n][k])
                    label_set[n] = true
                else
                    violin!([values2[i]], interest2[i], outliers=false, label = "", alpha = 0.5, color = colours[n][k])
                end
            end
            #Calculate means and store for later
            means = [(i ≤ length(interest3) && !isempty(interest3[i])) ? mean(interest3[i]) : NaN for i in eachindex(values2)]
            means2 = [(i ≤ length(interest2) && !isempty(interest2[i])) ? mean(interest2[i]) : NaN for i in eachindex(values2)]
            push!(means_truncated[n][k], means2)
            push!(means_full[n][k], means)

            #plot!(values2, means2, linecolor = colours[k], linewidth = 2, label = "means of all lower than $(upper_plotting_bound[k]) for "*models[k])
            scatter!(values2, means, markercolor = colours[n][k], markershape = :star5, linewidth = 2, label = "means overall for "*models[k]*names_distinction[n])
            scatter!(values2, means2, markercolor = colours[n][k], markershape = :circle, linewidth = 1, alpha = 0.7, label = "means of all lower than $(upper_plotting_bound[k]) for "*models[k]*names_distinction[n])
            #Plot untruncated means only in full one?
            #plot!(values2, means2, linecolor = :blue, linewidth = 2, label = "means")
            if spaced_accordingly
                plot!(xticks = values2, xrotation = 90)
            end
        end
    end
    plot!(legend=:outerbottom, legendcolumns=1)
    push!(per_model_plots[k],pl)
    display(pl)
end

#Overall means plots
overall_plots = Plots.Plot[]
pl2 = plot(xlabel = quant, ylabel = "means of "*lowercasefirst(name), title = ("Truncated means of "*lowercasefirst(name)*" for \n different "*lowercasefirst(quant)))
for k in eachindex(models)
    for n in 1:2
        if !isempty(means_truncated[n][k])
            plot!(values, means_truncated[n][k], linecolor = colours[n][k], linewidth = 2, label = "means of all lower than $(upper_plotting_bound[k]) for "*models[k]*names_distinction[n])
        end
    end
    plot!(xticks = values, xrotation = 75)
    plot!(legend=:outerbottom, legendcolumns=legendcolumns)
end
push!(overall_plots, pl2)
display(pl2)

if plot_separate && any(.!isempty.(means_truncated[1])) && any(.!isempty.(means_truncated[2]))
    for n in 1:2
        pl25 = plot(xlabel = quant, ylabel = "means of "*lowercasefirst(name), title = ("Truncated means of "*lowercasefirst(name)*" for \n different "*lowercasefirst(quant)*names_distinction[n]))
        for k in eachindex(models)
            if !isempty(means_truncated[n][k])
                plot!(values, means_truncated[n][k], linecolor = colours[n][k], linewidth = 2, label = "means of all lower than $(upper_plotting_bound[k]) for "*models[k]*names_distinction[n])
            end
        end
            plot!(xticks = values, xrotation = 75)
            plot!(legend=:outerbottom, legendcolumns=legendcolumns)
        push!(overall_plots, pl25)
        display(pl25)
    end
end

pl3 = plot(xlabel = uppercasefirst(quant), ylabel = "means of "*lowercasefirst(name), title = ("Overall means of "*lowercasefirst(name)*" for \n different "*lowercasefirst(quant)))
for k in eachindex(models)
    for n in 1:2
        if !isempty(means_truncated[n][k])
            plot!(values, means_full[n][k], linecolor = colours[n][k], linewidth = 2, label = "means of all for "*models[k]*names_distinction[n])
        end
    end
    plot!(xticks = values, xrotation = 75)
    plot!(legend=:outerbottom, legendcolumns=legendcolumns)
end
push!(overall_plots, pl3)
display(pl3)

if plot_separate && any(.!isempty.(means_full[1])) && any(.!isempty.(means_full[2]))
    for n in 1:2
        pl35 = plot(xlabel = quant, ylabel = "means of "*lowercasefirst(name), title = ("Overall means of "*lowercasefirst(name)*" for \n different "*lowercasefirst(quant)*names_distinction[n]))
        for k in eachindex(models)
            if !isempty(means_full[n][k])
                plot!(values, means_full[n][k], linecolor = colours[n][k], linewidth = 2, label = "means of all for "*models[k]*names_distinction[n])
            end
        end
            plot!(xticks = values, xrotation = 75)
            plot!(legend=:outerbottom, legendcolumns=legendcolumns)
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