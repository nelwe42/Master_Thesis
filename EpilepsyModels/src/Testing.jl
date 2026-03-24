using Pkg

println("Starting")
#Pkg.develop(path = ".//EpilepsyModels")
include("EpilepsyModels.jl")

using .EpilepsyModels
using ComponentArrays
using OptimizationOptimJL
using LineSearches
using DifferentialEquations
using Plots
using StaticArrays
using Random
using BenchmarkTools

println("Included")

#set seed
Random.seed!(42)

#Specify parameters for models
#PK Models
Input_θ_PKBasic = ComponentArray((k_el = 2.0, k_abs = 5.0, σ=0.2))
Input_θ_PKLEV = ComponentArray((k_abs = (24*3.5), c1 = (24*4.0), c2 = 0.25, c3 = 0.6, v1 = 29.7, v2 = 2.85, σ=0.2))
Input_θ_PKLEVNoAbsorption = ComponentArray((c1 = (24*4.0), c2 = 0.25, c3 = 0.6, v1 = 29.7, v2 = 2.85, σ=0.2))
Input_θ_PKCBZ = ComponentArray((k_abs = (24*0.45), c1 = (24*1.96), c2 = 1.73, c3 = 24*1.36, v1 = 164.0/75.0, σ=0.2))
Input_θ_PKVPA = ComponentArray((k_abs = (24*1.86), c1 = (24*15.6/1000), c2 = 0.748, c3 = 0.183, c4 = 0.898, v1 = 0.28, σ=0.2))
#Seizure Models
Input_θ_SeizureBasic = ComponentArray((a = 1.5, b = SA[0.05]))

Maxiters_optimiser = 200
Population_size = 2 #10 #20
wo_treatment = 3.0 #10.0
Obs_Duration = wo_treatment + 20.0 #40.0
PK_timepoints = wo_treatment:3.75:Obs_Duration
#logscale = ("σ",)
logscale = ("σ", "k_abs", "c1", "v1", "a")
solver_optim = LBFGS(linesearch = LineSearches.BackTracking())
ODE_options = (AutoTsit5(Rosenbrock23()),)
#ODE_options = (Rosenbrock23(),)

#Multistart settings (LHS) for robust optimisation from weak/default initial guesses.
#All bounds are in transformed space (i.e. log-scale for logscale parameters).
max_threads_simul = 5
Multistart_nstarts = 5
Multistart_seed = 42
Multistart_include_initial = true
bound_abs = nothing #100.0
optim_lower_bounds = nothing
optim_upper_bounds = nothing 
Variance_bound = log(1.0) #upper bounds will be reset accordingly after Input_θ is created below
Multistart_bounds = 10.0 #nothing

run_hessian = true
finite_diff_hessian = true
drug_appropriate_dosing = true
hierarchical_optimisation = true
plotting = false

#pk_model = PKBasic(θ=Input_θ_PKBasic)
#pk_model = PKLEV(θ=Input_θ_PKLEV)
pk_model = PKLEVNoAbsorption(θ=Input_θ_PKLEVNoAbsorption)
#pk_model = PKCBZ(θ=Input_θ_PKCBZ)
#pk_model = PKVPA(θ=Input_θ_PKVPA)
seizure_model = SeizureBasic(θ = Input_θ_SeizureBasic)
person_gen = BigFourPersonGenerator()
#dose_gen = BasicDoses(default_dose=500.0, times_per_day=2)
#dose_gen = PolyDoses(pk_model, default_dose=500.0)
#Create appropriate dose generator based on which pk_model was chosen
dose_distr = (d_VPA = (min = 150.0, avg_num = 5.0, max_num = 14), d_LEV = (min = 100.0, avg_num = 10.0, max_num = 30), s_LEV_unnormalised = (min = 100.0, avg_num = 10.0, max_num = 30),
                d_LTG = (min = 25.0, avg_num = 4.0, max_num = 24), d_CBZ = (min = 200.0, avg_num = 3.0, max_num = 8))
if (typeof(pk_model).name.wrapper in [PKLEV, PKLEVNoAbsorption, PKCBZ, PKVPA]) && drug_appropriate_dosing #add PKLTG here later
    info = dose_distr[pk_model.keys.d][1]
    dose_gen = PolyDosesRandom(pk_model, default_min_dose = info.min, default_avg_multiple_dose = info.avg_num, default_max_multiple_dose = info.max_num, times_per_day_first = 2)
else
    dose_gen = PolyDosesRandom(pk_model, default_min_dose = 100.0)
end
Input_θ = ComponentArray(PK = pk_model.θ, Seizure = seizure_model.θ)
mod = FullModel(pk_model, seizure_model, person_gen, dose_gen)
data = generate_data(mod, Population_size, Obs_Duration, timepoints = PK_timepoints, wo_treatment = wo_treatment, ODE_options = ODE_options)
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
test_mod = FullModel(typeof(pk_model).name.wrapper(), typeof(seizure_model).name.wrapper(), person_gen, dose_gen)
if hierarchical_optimisation
    #Normal optimisation
    estimate = optimise_hierarchical(test_mod, data, maxiters = Maxiters_optimiser, logscale = logscale, 
                        bound_abs = bound_abs, lower_upper = lower_upper_bounds, 
                        solver_optim = solver_optim, ODE_options = ODE_options)
    println("True Objective Value: ", get_negloglikelihood_evaluated_hierarchical(Input_θ, mod, data, logscale = logscale, ODE_options = ODE_options))
else
    #Multistart optimisation
    estimate = optimise(test_mod, data, maxiters = Maxiters_optimiser, logscale = logscale, solver_optim = solver_optim, ODE_options = ODE_options,
        bound_abs = bound_abs, lower_upper = lower_upper_bounds, 
        multistart = Multistart_nstarts, max_threads = max_threads_simul, multistart_seed = Multistart_seed,
        multistart_include_initial = Multistart_include_initial, multistart_bounds = Multistart_bounds)
    println("True Objective Value: ", get_negloglikelihood_evaluated(Input_θ, mod, data, logscale = logscale, ODE_options = ODE_options))
end
#True values for comparison
println("True θ: ", Input_θ)
#Show relative error
errors = deepcopy(Input_θ)
for i in eachindex(errors)
    errors[i] = abs(estimate.u[i] - Input_θ[i])/abs(Input_θ[i])
end
println("Relative errors: ", errors)
#Testing out hessian confidence intervals if flag is set
if run_hessian && SciMLBase.successful_retcode(estimate.retcode)
    CI = EpilepsyModels.inverse_hessian(estimate.u, mod, data, logscale=logscale, finite_not_forward=finite_diff_hessian, ODE_options = ODE_options)
    #CI = EpilepsyModels.inverse_hessian(Input_θ, mod, data, logscale=logscale, ODE_options = ODE_options)
    println("Confidence Intervals Inverse Hessian:", CI)
end

if plotting
    #Plot PK behavior (for each drug)
    names = EpilepsyModels.get_keys_PK(mod.pk_model)
    i = 1 #index of person for which plotting is done
    sol = EpilepsyModels.solve_PK(mod.pk_model, mod.pk_model.θ, data[i], endpoint = Obs_Duration, options = ODE_options)
    for s in names.s
        pl = plot(sol, idxs = s, label="Concentration $(s)", 
            xlabel="Time", ylabel="Amount", title="PK Trajectory of $(s) for person $(i)")
        x_values = [measurement.timepoint for measurement in data[i].measurements if (measurement.state[2] == s)]
        y_values = [measurement.measurement for measurement in data[i].measurements if (measurement.state[2] == s)]
        plot!(x_values, y_values, seriestype = :scatter, mc = :purple, label = "")
        #add estimate plot
        Estimate_θ = estimate.u
        sol2 = EpilepsyModels.solve_PK(mod.pk_model, Estimate_θ.PK, data[i], endpoint = Obs_Duration, options = ODE_options)
        pl = plot!(sol2, idxs = s, label="Estimated concentration $(s)", linecolor = :red)

        display(pl)
    end

    #Plot all trajectories in one
    sols = [EpilepsyModels.solve_PK(mod.pk_model, mod.pk_model.θ, data[i], endpoint = Obs_Duration, options = ODE_options) for i in 1:Population_size]
    for s in names.s
        pl = plot(sols[1], idxs = s, label="Concentration $(s)", linecolor = :blue,
            xlabel="Time", ylabel="Amount", title="PK Trajectory of $(s)")
        for i in 2:Population_size
            plot!(sols[i], idxs = s, label = "", linecolor = :blue)
        end
        for i in 1:Population_size
            x_values = [measurement.timepoint for measurement in data[i].measurements if (measurement.state[2] == s)]
            y_values = [measurement.measurement for measurement in data[i].measurements if (measurement.state[2] == s)]
            plot!(x_values, y_values, seriestype = :scatter, mc = :purple, label = "")
        end
        #add estimate plots
        Estimate_θ = estimate.u
        sols2 = [EpilepsyModels.solve_PK(mod.pk_model, Estimate_θ.PK, data[i], endpoint = Obs_Duration, options = ODE_options) for i in 1:Population_size]
        plot!(sols2[1], idxs = s, label="Estimated concentration $(s)", linecolor = :red)
        for i in 2:Population_size
            plot!(sols2[i], idxs = s, label = "", linecolor = :red)
        end
        plot!(legend=:outerbottom, legendcolumns=2)
        display(pl)
    end
end

println("Done")