using Pkg

#Pkg.instantiate()
println("Starting")
#Pkg.develop(path = ".//EpilepsyModels")
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

#This will redirect output to txt file, not including error messages/warnings
#path = "."
path = "/home/s6newell_hpc"
#open(joinpath(path,"output.txt"), "w") do io
#open(joinpath(path,"errors.txt"), "w") do io_err
#redirect_stdout(io) do
#redirect_stderr(io_err) do

println("Included")

#try

#set seed
Random.seed!(42)

#Specify parameters for models
#PK Models
Input_θ_PKBasic = ComponentArray((k_el = 2.0, k_abs = 5.0, σ=0.2))
Input_θ_PKLEV = ComponentArray((k_abs = (24*3.5), c1 = (24*4.0), c2 = 0.25, c3 = 0.122, v1 = 29.7, v2 = 2.85, σ=0.2))
Input_θ_PKLEVNoAbsorption = ComponentArray((c1 = (24*4.0), c2 = 0.25, c3 = 0.122, v1 = 29.7, v2 = 2.85, σ=0.2))
Input_θ_PKCBZ = ComponentArray((k_abs = (24*0.45), c1 = (24*1.96), c2 = 1.73, c3 = 24*1.36, v1 = 164.0/75.0, σ=0.2))
Input_θ_PKVPA = ComponentArray((k_abs = (24*1.86), c1 = (24*0.577), c2 = 0.535, c3 = 0.875, v1 = 0.28, σ=0.2))
Input_θ_PKLTG = ComponentArray((k_abs = (24*1.96), c1 = (24*2.4), c2 = 0.938, c3 = 110*0.00328, c4 = 0.34, v1 = 2.14, σ=0.2))
Input_θ_PKBigFour = ComponentArray((k_abs_LTG = (24*1.96), c1_LTG = (24*2.4), c2_LTG = 0.938, c3_LTG = 110*0.00328, c4_LTG = 0.34, c_Inh_LTG = (1-0.579), c_Ind_LTG = (1+0.546), v1_LTG = 2.14, σ_LTG=0.2,
            k_abs_VPA = (24*1.86), c1_VPA = (24*0.577), c2_VPA = 0.535, c3_VPA = 0.875, c_Ind_VPA = 1.22, v1_VPA = 0.28, σ_VPA=0.2,
            k_abs_CBZ = (24*0.45), c1_CBZ = (24*1.96), c2_CBZ = 1.73, c3_CBZ = 24*1.36, v1_CBZ = 164.0/75.0, σ_CBZ=0.2,
            k_abs_LEV = (24*3.5), c1_LEV = (24*4.0), c2_LEV = 0.25, c3_LEV = 0.122, c_Inh_LEV = 0.812, c_Ind_LEV = 1.22, v1_LEV = 29.7, v2_LEV = 2.85, σ_LEV=0.2))
#Seizure Models
Input_θ_SeizureBasic_one = ComponentArray((a = 4.0, b = SA[0.2]))
Input_θ_SeizureBasic_four = ComponentArray((a = 4.0, b = SA[0.2, 0.2, 0.2, 0.05]))
#Input_θ_SeizureNegativeBinomial = ComponentArray((a = -1.923, o = 1.128, prev = 0.731, b = SA[0.2]))
Input_θ_SeizureNegativeBinomial = ComponentArray((a = log(4.0), o = 1.128, prev = 0.731, b = SA[0.2]))

Maxiters_optimiser = 200
#Max_Time = 5*60.0 #
Max_Time = 10.0*60*60 #4.0*60*60 + 30.0*60 #maximal optimisertime in seconds
Population_size = 5 #10 #20
wo_treatment = 0.0 #10.0
Obs_Duration = wo_treatment + 30.0 #40.0
PK_timepoints = wo_treatment:3.75:Obs_Duration
Seizure_timepoints = 0.0:1.0:Obs_Duration
no_counts_seizure = false
#logscale = ("σ",)
#logscale = ("σ","v1")
logscale = ("σ", "k_abs", "c1", "v1", "a")
#logscale = ("σ_LEV", "k_abs_LEV", "c1_LEV", "v1_LEV", "σ_LTG", "k_abs_LTG", "c1_LTG", "v1_LTG","σ_CBZ", "k_abs_CBZ", "c1_CBZ", "v1_CBZ", "σ_VPA", "k_abs_VPA", "c1_VPA", "v1_VPA", "a")
#logscale = ("σ", "k_abs", "c1", "c3", "v1", "a") 
#logscale = ("σ", "c1", "v1", "a")
println("logscale = ", logscale)
solver_optim = LBFGS(linesearch = LineSearches.BackTracking())
#solver_optim = BBO_adaptive_de_rand_1_bin_radiuslimited()
ODE_options = (AutoTsit5(Rosenbrock23()),)
#ODE_options = (Rosenbrock23(),)

#Multistart settings (LHS) for robust optimisation from weak/default initial guesses.
#All bounds are in transformed space (i.e. log-scale for logscale parameters).
max_threads_simul = 5
Multistart_nstarts = 1
Multistart_seed = 42
Multistart_include_initial = true
bound_abs = nothing #100.0
optim_lower_bounds = nothing
optim_upper_bounds = nothing
Variance_bound = log(1.0) #upper bounds will be reset accordingly after Input_θ is created below
Multistart_bounds = 20.0 #nothing
fail_hard = false

run_hessian = false
sandwich = true
finite_diff_hessian = false
drug_appropriate_dosing = false
hierarchical_optimisation = false
plotting = false
optimisation_trace = true
show_trace = true

#pk_model = PKBasic(θ=Input_θ_PKBasic)
pk_model = PKLEV(θ=Input_θ_PKLEV)
#pk_model = PKLEVNoAbsorption(θ=Input_θ_PKLEVNoAbsorption)
#pk_model = PKCBZ(θ=Input_θ_PKCBZ)
#pk_model = PKVPA(θ=Input_θ_PKVPA)
#pk_model = PKLTG(θ=Input_θ_PKLTG)
#pk_model = PKBigFour(θ = Input_θ_PKBigFour)
#Set b in seizure_basic according to pk model (different daily exposures), for VPA 0.2 is too high
if (typeof(pk_model).name.wrapper in [PKVPA])
    Input_θ_SeizureBasic.b = SA[0.05]
end
if (typeof(pk_model).name.wrapper in [PKBigFour])
    seizure_model = SeizureBasic(θ = Input_θ_SeizureBasic_four)
else
    seizure_model = SeizureBasic(θ = Input_θ_SeizureBasic_one)
end
#seizure_model = SeizureNegativeBinomial(θ = Input_θ_SeizureNegativeBinomial)
person_gen = BigFourPersonGenerator()
#dose_gen = BasicDoses(default_dose=500.0, times_per_day=2)
#dose_gen = PolyDoses(pk_model, default_dose=500.0)
#Create appropriate dose generator based on which pk_model was chosen
dose_gen = PolyDosesRandom(pk_model, drug_appropriate_dosing)
Input_θ = ComponentArray(PK = pk_model.θ, Seizure = seizure_model.θ)
mod = FullModel(pk_model, seizure_model, person_gen, dose_gen)
data = generate_data(mod, Population_size, Obs_Duration, timepoints_PK = PK_timepoints, timepoints_seizure = Seizure_timepoints, wo_treatment = wo_treatment, just_Bool = no_counts_seizure, ODE_options = ODE_options)
println("Generated")


#Set bounds on sigma, ensure both or neither of lower/upper bounds are nothing
if isnothing(optim_upper_bounds) && (!isnothing(Variance_bound) || !isnothing(optim_lower_bounds))
    upper_bounds = ComponentArray(fill(Inf, length(Input_θ)), getaxes(Input_θ))
else 
    upper_bounds = optim_upper_bounds
end
if isnothing(optim_lower_bounds) && (!isnothing(optim_upper_bounds) || !isnothing(upper_bounds))
    lower_bounds = ComponentArray(fill(-Inf, length(Input_θ)), getaxes(Input_θ))
else 
    lower_bounds = optim_lower_bounds
end
if !isnothing(Variance_bound) 
    #upper_bounds.PK.σ = min(Variance_bound, upper_bounds.PK.σ)
    labels_PK = labels(upper_bounds.PK)
    #If multiple σ's denoted like σ_drugname
    for i in eachindex(labels_PK)
        if occursin("σ", labels_PK[i])
            upper_bounds[i] = min(Variance_bound, upper_bounds[i])
        end
    end
end
lower_upper_bounds = (lower_bounds, upper_bounds)
if isnothing(lower_bounds)
    lower_upper_bounds = nothing
end

#create test mod of same types as true ones
test_mod = FullModel(typeof(pk_model).name.wrapper(), typeof(seizure_model).name.wrapper(length(pk_model.keys.S)), person_gen, dose_gen)
println("PK start: ", test_mod.pk_model.θ)
if hierarchical_optimisation
    #Hierarchical optimisation
    estimate = optimise_hierarchical(test_mod, data, maxiters = Maxiters_optimiser, logscale = logscale, 
                        bound_abs = bound_abs, lower_upper = lower_upper_bounds, objective_fail_hard=fail_hard, store_trace = optimisation_trace,
                        solver_optim = solver_optim, ODE_options = ODE_options)
    println("True Objective Value: ", get_negloglikelihood_evaluated_hierarchical(Input_θ, mod, data, logscale = logscale, ODE_options = ODE_options))
else
    #Multistart optimisation
    estimate = optimise(test_mod, data, maxiters = Maxiters_optimiser, maxtime = Max_Time, logscale = logscale, solver_optim = solver_optim, ODE_options = ODE_options,
        bound_abs = bound_abs, lower_upper = lower_upper_bounds, objective_fail_hard=fail_hard, store_trace = optimisation_trace,
        multistart = Multistart_nstarts, max_threads = max_threads_simul, multistart_seed = Multistart_seed,
        multistart_include_initial = Multistart_include_initial, multistart_bounds = Multistart_bounds)
    println("True Objective Value: ", get_negloglikelihood_evaluated(Input_θ, mod, data, logscale = logscale, ODE_options = ODE_options))
end
#True values for comparison
println("True θ: ", Input_θ)
#Show relative error
errors_rel = deepcopy(Input_θ)
errors_abs = deepcopy(Input_θ)
for i in eachindex(errors_rel)
    errors_rel[i] = abs(estimate.u[i] - Input_θ[i])/abs(Input_θ[i])
end
for i in eachindex(errors_abs)
    errors_abs[i] = abs(estimate.u[i] - Input_θ[i])
end
println("Relative errors: ", errors_rel)
println("Absolute errors: ", errors_abs)
#Testing out hessian confidence intervals if flag is set
if run_hessian && SciMLBase.successful_retcode(estimate.retcode)
    CI = EpilepsyModels.inverse_hessian(estimate.u, mod, data, logscale=logscale, finite_not_forward=finite_diff_hessian, sandwich = sandwich, ODE_options = ODE_options)
    #CI = EpilepsyModels.inverse_hessian(Input_θ, mod, data, logscale=logscale, ODE_options = ODE_options)
    println("Confidence Intervals: ", CI)
end

if optimisation_trace && show_trace
    if !(hierarchical_optimisation)
        println(estimate.raw.original.trace)
    else
        println("PK trace: ")
        println(estimate.estimate_PK.original.trace)
        println("Seizure trace: ")
        println(estimate.estimate_Seizure.original.trace)
    end
end

if plotting && SciMLBase.successful_retcode(estimate.retcode)
    #Plot fit for specified individuals
    individuals = [1]
    time_seizures = (0,10)
    time_pk = (0.0, Obs_Duration)
    plots = plot_fit(mod, data, true_param = Input_θ, estimate_param = estimate.u, individuals = individuals, endpoint = Obs_Duration, time_pk = time_pk, time_seizures = time_seizures, options = ODE_options)
end

plot_change = false
#Plotting change for k_abs/other param specified through index
if plot_change
indices_interest = [5] #[1,2,3,4,5,6,8,9]#[3,4,5] 
for j in indices_interest
#for multi in 1:10
    point, name_point = ComponentArray(PK = typeof(pk_model).name.wrapper().θ, Seizure = typeof(seizure_model).name.wrapper().θ), "Default_Start"
    #point[2], name_point = Input_θ[2], "Default_Start_true_c1"
    #point[j] = multi #24*0.5*multi
    #name_point = "Default_Start_with_$(labels(point)[j])=$(point[j])"
    point[2] = 3*24.0
    name_point = "Default_Start_with_c1=72.0"
    #point, name_point = Input_θ, :True_Values
    #point, name_point = estimate.u, :Estimate
    #point, name_point = ComponentArray(PK = test_mod.pk_model.θ, Seizure = test_mod.seizure_model.θ), :Set_Default_Start
    #index of parameter to consider, name
    index, index_name = 1, :k_abs
    #index, index_name = 2, :c1
    #index, index_name = j, "$(labels(point)[j])"
    #index, index_name = 5, "v1"
    θ_use = deepcopy(point)
    names = mod.pk_model.keys
    sys = EpilepsyModels.create_ode_system(mod.pk_model)
    problems = Tuple(EpilepsyModels.create_problem(mod.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time[2])) for person in data)
    indices_θ = [ModelingToolkit.parameter_index(sys, x).idx for x in keys(θ_use.PK)]
    EpilepsyModels.partial_transform_to_logscale!(θ_use, logscale = logscale)
    p = (m = mod, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ)
    function negloglikeli(x::AbstractFloat)
        θ_vary = copy(θ_use)
        θ_vary[index] = x
        return EpilepsyModels.get_negloglikelihood(θ_vary, p)
    end
    plot_for = [20.0, 120.0] #plotting range in untransformed space
    if String(index_name) in logscale
        plot_for .= log.(plot_for)
    end
    x = range(plot_for[1], plot_for[2], length=100)
    y = negloglikeli.(x)
    pl = plot(x, y, title="$(name_point) \n logscale = $(logscale)", xlabel="$(index_name)", ylabel="Negloglikelihood", label = "$(typeof(pk_model).name.wrapper)")
    display(pl)
#end
end
end

#For adding custom xticks (as vertical lines) to plot
#plot!([log(24.0), log(1.5*24.0), log(2*24.0), log(3*24.0)], seriestype="vline", xticks = ([log(24.0), log(1.5*24.0), log(2*24.0), log(3*24.0)],["log(1.0*24)","log(1.5*24)", "log(2.0*24)", "log(3.0*24)"]), linecolor = "grey", label="")
#plot!(xticks = ([log(24.0), log(1.5*24.0), log(2*24.0), log(3*24.0)],["log(1.0*24)","log(1.5*24)", "log(2.0*24)", "log(3.0*24)"]))

println("Done")

#catch e
#   @warn e
#end

#end
#end
#end
#end