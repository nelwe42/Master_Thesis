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
using FileWatching

#This will redirect output to txt file, not including error messages/warnings
path = "/home/s6newell_hpc"
lock_path_output = "./output.txt.lock"

#Can e.g. use parsed job array id as seed here
if !isempty(ARGS)
    parsed = parse(Int, ARGS[1])
else
    parsed = 42
end

#set seed
Random.seed!(parsed)

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
            k_abs_LEV = (24*3.5), c1_LEV = (24*4.0), c2_LEV = 0.25, c3_LEV = 0.122, c_Inh_LEV = 0.812, c_Ind_LEV = 1.09, v1_LEV = 29.7, v2_LEV = 2.85, σ_LEV=0.2))
#Seizure Models
base_rate = 4.0
Input_θ_SeizureBasic_one = ComponentArray((a = base_rate, b = SA[0.2]))
Input_θ_SeizureBasic_four = ComponentArray((a = base_rate, b = SA[base_rate/7, base_rate/25, base_rate/20, base_rate/120]))
#Input_θ_SeizureNegativeBinomial = ComponentArray((a = -1.923, o = 1.128, prev = 0.731, b = SA[0.2]))
Input_θ_SeizureNegativeBinomial = ComponentArray((a = log(4.0), o = 1.128, prev = 0.731, b = SA[0.2]))
Input_θ_SeizureVPA = ComponentArray((a = 6.1, a1 = 1.0, a2 = 1.8, b1 = 13.3, b2 = 2.4))

Maxiters_optimiser = 200
Samples_per_chain = 2000
Adaptation_steps = 1000
Max_Time = 7*60.0*60.0
#Max_Time = 23.5*60*60 #4.0*60*60 + 30.0*60 #maximal optimisertime in seconds
Population_size = 10 #5 #20 #10 #20
wo_treatment = 0.0 #10.0
Obs_Duration = wo_treatment + 20.0 #40.0
PK_timepoints = wo_treatment:3.75:Obs_Duration
Seizure_timepoints = 0.0:1.0:Obs_Duration
no_counts_seizure = false
#logscale = ("σ",)
#logscale = ("σ","v1")
#logscale = ("σ", "k_abs", "c1", "v1", "a")
logscale = ("σ", "k_abs", "c1", "v1")
#logscale = ("σ_LEV", "k_abs_LEV", "c1_LEV", "v1_LEV", "σ_LTG", "k_abs_LTG", "c1_LTG", "v1_LTG","σ_CBZ", "k_abs_CBZ", "c1_CBZ", "v1_CBZ", "σ_VPA", "k_abs_VPA", "c1_VPA", "v1_VPA", "a")
#logscale = ("σ", "k_abs", "c1", "c3", "v1", "a") 
#logscale = ("σ", "c1", "v1", "a")
solver_optim = LBFGS(linesearch = LineSearches.BackTracking())
#solver_optim = BBO_adaptive_de_rand_1_bin_radiuslimited()
sampler = NUTS(0.8) #nothing
sampling_options = (verbose = false, progress = false) #limits output
ODE_options = (AutoTsit5(Rosenbrock23()),)

#Multistart settings (LHS) for robust optimisation from weak/default initial guesses.
#All bounds are in transformed space (i.e. log-scale for logscale parameters).
run_count = 1
max_threads_simul = Threads.nthreads()
max_threads_runs = min(run_count, max_threads_simul)
max_threads_simul = Int(max(floor(max_threads_simul/run_count),1))
Multistart_nstarts = 2
prefilter = 10
Multistart_seed = 42
Multistart_include_initial = true
custom_starts = nothing
bound_abs = nothing #100.0
optim_lower_bounds = nothing
optim_upper_bounds = nothing
Variance_bound = nothing #log(1.0) #upper bounds will be reset accordingly after Input_θ is created below
Multistart_bounds = nothing #25.0
fail_hard = false

run_CI = true
confidence = 0.95
sandwich = true
finite_diff_hessian = false
drug_appropriate_dosing = true
hierarchical_optimisation = false
sampling = false
plotting = false
show_original = true

#pk_model = PKBasic(θ=Input_θ_PKBasic)
pk_model = PKLEV(θ=Input_θ_PKLEV)
#pk_model = PKLEVNoAbsorption(θ=Input_θ_PKLEVNoAbsorption)
#pk_model = PKCBZ(θ=Input_θ_PKCBZ)
#pk_model = PKVPA(θ=Input_θ_PKVPA)
#pk_model = PKLTG(θ=Input_θ_PKLTG)
#pk_model = PKBigFour(θ = Input_θ_PKBigFour)
#Set b in seizure_basic according to pk model (different daily exposures), for VPA 0.2 is too high
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
#seizure_model = SeizureMult(pk_model, base_rate = base_rate, default_treat_eff = 0.2)
#seizure_model = SeizureVPA(θ = Input_θ_SeizureVPA)
#seizure_model = SeizureNegativeBinomial(θ = Input_θ_SeizureNegativeBinomial)

person_gen = BigFourPersonGenerator()
#dose_gen = BasicDoses(default_dose=500.0, times_per_day=2)
#dose_gen = PolyDoses(pk_model, default_dose=500.0)
#Create appropriate dose generator based on which pk_model was chosen
dose_gen = PolyDosesRandom(pk_model, drug_appropriate_dosing)
#dose_gen = BigFourDoses()
if seizure_model isa SeizureVPA
    dose_distr = (d_VPA = (min = 150.0, avg_num = 8.0, max_num = 14), d_CBZ = (min = 200.0, avg_num = 3.0, max_num = 8))
    distr_first = (d_VPA = 1.0, d_CBZ = 0.0)
    distr_second = (d_VPA = 0.0, d_CBZ = 1.0)
    dose_gen = PolyDosesRandom(dose_distr, distr_first, distr_second; prob_second=0.5, times_per_day_first=2, times_per_day_second=2, assign_not_supported = true)
    #dose_gen = BigFourDoses(order_male = ((:d_VPA,:d_CBZ), (:d_VPA,:d_CBZ)), order_female = ((:d_VPA,:d_CBZ), (:d_VPA,:d_CBZ)), prob_second = 1.0)
end
Input_θ = ComponentArray(PK = pk_model.θ, Seizure = seizure_model.θ)
mod = FullModel(pk_model, seizure_model, person_gen, dose_gen)

#Set data generating function
data() = generate_data(mod, Population_size, Obs_Duration, timepoints_PK = PK_timepoints, timepoints_seizure = Seizure_timepoints, wo_treatment = wo_treatment, max_threads = max_threads_simul, just_Bool = no_counts_seizure, ODE_options = ODE_options)
modifications = ((2, Normal(0, Input_θ[2]/4)),(label2index(Input_θ,"Seizure.a")[1], Normal(0,Input_θ.Seizure.a/6)))
data_modified() = generate_data_modified(mod, Population_size, Obs_Duration, update_reg = 5.0, modifications = modifications, timepoints_PK = PK_timepoints, timepoints_seizure = Seizure_timepoints, max_threads = max_threads_simul, just_Bool = no_counts_seizure, wo_treatment = wo_treatment, ODE_options = ODE_options)
data_updating() = generate_data_updating(mod, Population_size, Obs_Duration, update_reg = 5.0, timepoints_PK = PK_timepoints, timepoints_seizure = Seizure_timepoints, max_threads = max_threads_simul, just_Bool = no_counts_seizure, wo_treatment = wo_treatment, ODE_options = ODE_options)


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

#Specify estimator function
if hierarchical_optimisation
    #Hierarchical optimisation
    estimate(new_mod, data) = optimise_hierarchical(new_mod, data, maxiters = Maxiters_optimiser, logscale = logscale, 
                        bound_abs = bound_abs, lower_upper = lower_upper_bounds, objective_fail_hard=fail_hard, store_trace = false,
                        printing = false, solver_optim = solver_optim, ODE_options = ODE_options, 
                        run_CI = run_CI, confidence = confidence, finite_not_forward = finite_diff_hessian, sandwich = sandwich)
    eval(data) = get_negloglikelihood_evaluated_hierarchical(Input_θ, mod, data, logscale = logscale, ODE_options = ODE_options)
elseif sampling
    estimate(new_mod, data) = optimise_sampled(deepcopy(new_mod), data, per_chain=Samples_per_chain, nadapts = Adaptation_steps, 
                        bound_abs = bound_abs, lower_upper = lower_upper_bounds, objective_fail_hard=fail_hard, multistart = Multistart_nstarts, 
                        prefilter = prefilter, custom_starts = custom_starts, max_threads = max_threads_simul, multistart_seed = Multistart_seed, 
                        multistart_include_initial = Multistart_include_initial, multistart_bounds = Multistart_bounds, printing = false, 
                        run_CI = run_CI, confidence = confidence, sampler = sampler, sampling_options = sampling_options) 
    eval(data) = get_negloglikelihood_evaluated(Input_θ, mod, data, logscale = logscale, ODE_options = ODE_options)
else
    estimate(new_mod, data) = optimise(new_mod, data, maxiters = Maxiters_optimiser, maxtime = Max_Time, logscale = logscale, solver_optim = solver_optim, ODE_options = ODE_options,
        bound_abs = bound_abs, lower_upper = lower_upper_bounds, objective_fail_hard=fail_hard, store_trace = false,
        multistart = Multistart_nstarts, prefilter = prefilter, custom_starts = custom_starts, max_threads = max_threads_simul, multistart_seed = Multistart_seed,
        multistart_include_initial = Multistart_include_initial, multistart_bounds = Multistart_bounds, printing = false,
        run_CI = run_CI, confidence = confidence, finite_not_forward = finite_diff_hessian, sandwich = sandwich)
    eval(data) = get_negloglikelihood_evaluated(Input_θ, mod, data, logscale = logscale, ODE_options = ODE_options)
end

try

results = multi_data_run(mod, data, estimate, eval, run_count=run_count, max_threads_runs=max_threads_runs)


#create lockfile to ensure no multiwriting
mkpidlock(lock_path_output; stale_age=30, wait=true) do
open(joinpath(path,"output.txt"), "a") do io
redirect_stdout(io) do
    println("My id is ", parsed)
    println()
    println("True θ: ", Input_θ)
    println()
    
#Print everything, including possibly originals
for a in keys(results)
    if a == :estimates
        println("Estimates:")
        for estimate in results.estimates
            println(estimate.u)
            if show_original
                if hierarchical_optimisation
                    #println()
                    println("PK output:")
                    println(estimate.estimate_PK.original)
                    #println()
                    println("Seizure output:")
                    println(estimate.estimate_Seizure.original)
                elseif !sampling
                    #println()
                    println(estimate.raw.original)
                end
            end
        end
        #println()
    elseif !(a==:datas)
        println("$(a):")
        println(results[a])
        #println()
    end
end
println()
end
end
end


catch e
    mkpidlock(lock_path_output; stale_age=30, wait=true) do
    open(joinpath(path,"output.txt"), "a") do io_err
    redirect_stderr(io_err) do
        @warn "At id $(parsed) encountered error: " e
    end
    end
    end
end