using ComponentArrays
using Plots
using StatsPlots
using Distributions

#files to read data from, relative to 
files = ["./PK_reg/CBZ_075.txt", "./PK_reg/CBZ_10.txt", "./PK_reg/CBZ_375.txt", "./PK_reg/CBZ_70.txt"]
#corresponding values of interest
values = [0.75, 1, 3.75, 7]

upper_outlier_bound = 1.1
spaced_accordingly = false

estimates_all = []
abs_errors_all = []
rel_errors_all = []
mean_squared_errors_all = []
rel_squared_errors_all = []
times_all = []
obj_diffs_all = []
CIs_all = []
for file in files
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
    push!(estimates_all, estimates)
    push!(abs_errors_all, abs_errors)
    push!(rel_errors_all, rel_errors)
    push!(mean_squared_errors_all, mean_squared_errors)
    push!(rel_squared_errors_all, rel_squared_errors)
    push!(times_all, times)
    push!(obj_diffs_all, obj_diffs)
    push!(CIs_all, CIs)
end

if spaced_accordingly
    values2 = deepcopy(values)
else
    values2 = String.(Symbol.(values))
end
#Mention how handle outliers
rel_squared_errors_all2 = deepcopy(rel_squared_errors_all)
for i in eachindex(rel_squared_errors_all2)
    rel_squared_errors_all2[i] = [rel for rel in rel_squared_errors_all2[i] if rel <= upper_outlier_bound]
end
pl = plot(xlabel = "Regularity", ylabel = "Relative squared error", title = "Relative squared errors for different \n regularities of PK measurements")
for i in eachindex(values)
    violin!([values2[i]], rel_squared_errors_all2[i], outliers=false, label = "", alpha = 0.5, color = :blue)
end
means = [mean(rel) for rel in rel_squared_errors_all2]
plot!(values2, means, linecolor = :blue, linewidth = 2, label = "mean")
if spaced_accordingly
    plot!(xticks = values2, xrotation = 90)
end

display(pl)
#print outliers
for i in eachindex(rel_squared_errors_all)
    println("Outliers for $(values[i]): ", [rel for rel in rel_squared_errors_all[i] if rel > upper_outlier_bound])
end