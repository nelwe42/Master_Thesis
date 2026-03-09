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

#Specify parameters
#Input_θ = ComponentArray((PK = ComponentArray((k_el = 2.0, k_abs = 5.0, σ=0.2)), 
#           Seizure = ComponentArray((a = 4, b = [-0.05]))))
Input_θ = ComponentArray((PK = ComponentArray((k_abs = (24*3.5), c1 = (24*4.0), c2 = 0.25, c3 = 0.6, v1 = 29.7, v2 = 2.85, σ=1.0)), 
           Seizure = ComponentArray((a = 4, b = SA[-0.05]))))
#Input_θ = ComponentArray((PK =ComponentArray((k_abs = 1.0, k_el = 1.0, σ = 0.1)), 
#            Seizure = ComponentArray((a = 4, b = SA[-0.05]))))
#Input_θ = ComponentArray((PK = ComponentArray((k_abs = (24*0.45), c1 = (24*1.96), c2 = 1.73, c3 = 24*1.36, v1 = 164.0/75.0, σ=1.0)), 
#           Seizure = ComponentArray((a = 4, b = SA[-0.05]))))
#Input_θ = ComponentArray((PK = ComponentArray((k_abs = (24*1.86), c1 = (24*15.6/1000), c2 = 0.748, c3 = 0.183, c4 = 0.898, v1 = 0.28, σ=1.0)), 
#	           Seizure = ComponentArray((a = 4, b = SA[-0.05]))))
Maxiters_optimiser = 100
Population_size = 2 #10 #20
wo_treatment = 0.0 #3.0 #10.0
Obs_Duration = wo_treatment + 20.0 #+40.0
PK_timepoints = wo_treatment:3.75:Obs_Duration
logscale = ("k_abs", "c1", "c2", "c3", "c4", "v1", "σ")
solver_optim = LBFGS(linesearch = LineSearches.BackTracking())
#ODE_options = (AutoTsit5(Rosenbrock23()),)
ODE_options = (Rosenbrock23(),)

#Multistart settings (LHS) for robust optimisation from weak/default initial guesses.
#All bounds are in transformed space (i.e. log-scale for logscale parameters).
Multistart_nstarts = 12
Multistart_seed = 42
Multistart_include_initial = true
Multistart_bound_abs = 10.0
Multistart_bounds = nothing

#Hessian via ForwardDiff + FiniteDiff is expensive; disable by default for routine runs.
Run_hessian = false

pk_model = PKLEV(θ=Input_θ.PK)
#pk_model = PKCBZ(θ=Input_θ.PK)
#pk_model = PKVPA(θ=Input_θ.PK)
seizure_model = SeizureBasic(θ = Input_θ.Seizure)
person_gen = BigFourPersonGenerator()
#dose_gen = BasicDoses(default_dose=500.0, times_per_day=2)
#dose_gen = PolyDoses(pk_model, default_dose=500.0)
dose_gen = PolyDosesRandom(pk_model, default_min_dose = 100.0)
mod = FullModel(pk_model, seizure_model, person_gen, dose_gen)
data = generate_data(mod, Population_size, Obs_Duration, timepoints = PK_timepoints, wo_treatment = wo_treatment, ODE_options = ODE_options)
println("Generated")

#create test mod of same types as true ones
test_mod = FullModel(typeof(pk_model).name.wrapper(), typeof(seizure_model).name.wrapper(), person_gen, dose_gen)
estimate = optimise(test_mod, data, maxiters = Maxiters_optimiser, logscale = logscale, solver_optim = solver_optim, ODE_options = ODE_options,
    bound_abs = Multistart_bound_abs, multistart = Multistart_nstarts, multistart_seed = Multistart_seed,
    multistart_include_initial = Multistart_include_initial, multistart_bounds = Multistart_bounds,
    objective_fail_hard=true)
#test out starting also in true value
#=
estimate = optimise(test, data, maxiters = Maxiters_optimiser, logscale = logscale, solver_optim = solver_optim, ODE_options = ODE_options,
    bound_abs = Multistart_bound_abs, multistart = Multistart_nstarts, multistart_seed = Multistart_seed,
    multistart_include_initial = Multistart_include_initial, multistart_bounds = Multistart_bounds,
    objective_fail_hard=true)
=#
println("Multistart: best start ", estimate.multistart_best_start, " of ", estimate.multistart_nstarts)
#True values for comparison
println("True Objective Value: ", get_negloglikelihood_evaluated(Input_θ, mod, data, logscale = logscale, ODE_options = ODE_options))
println("True θ: ", Input_θ)
#Testing out hessian
if Run_hessian && SciMLBase.successful_retcode(estimate.retcode)
    CI = EpilepsyModels.inverse_hessian(estimate.u, mod, data, logscale=logscale, ODE_options = ODE_options)
    println("Confidence Intervals Inverse Hessian:", CI)
elseif !Run_hessian
    println("Skipping inverse_hessian because Run_hessian=false")
else
    println("Skipping inverse_hessian because optimisation retcode was: ", estimate.retcode)
end

#Plot PK behavior (for each drug)
names = EpilepsyModels.get_keys_PK(mod.pk_model)
i = 1 #index of person for which plotting is done
sol = EpilepsyModels.solve_PK(mod.pk_model, mod.pk_model.θ, data[i], endpoint = Obs_Duration, options = ODE_options)
for s in names.s
    pl = plot(sol, idxs = s, labels=["Concentration $(s)"], 
        xlabel="Time", ylabel="Amount", title="PK Trajectory of $(s)")
    x_values = [measurement.timepoint for measurement in data[i].measurements if (measurement.state[2] == s)]
    y_values = [measurement.measurement for measurement in data[i].measurements if (measurement.state[2] == s)]
    plot!(x_values, y_values, seriestype = :scatter, mc = :purple, label = "")
    #add estimate plot
    if SciMLBase.successful_retcode(estimate.retcode)
        Estimate_θ = estimate.u
        sol2 = EpilepsyModels.solve_PK(mod.pk_model, Estimate_θ.PK, data[i], endpoint = Obs_Duration, options = ODE_options)
        pl = plot!(sol2, idxs = s, labels=["Estimated concentration $(s)"], linecolor = :red)
    end

    display(pl)
end

println("Done")

#testing callback daily dose
#=
dosing_test = [(t = 0.0, dose = 10, state = :d_CBZ), (t = 0.5, dose = 20, state = :d_CBZ), (t = 3.0, dose = 50, state = :d_CBZ)]
test_person = EpilepsyModels.Person(dosing = dosing_test)
sol = EpilepsyModels.solve_PK(mod.pk_model, mod.pk_model.θ, test_person, endpoint = 5.0, options = ODE_options)
pl = plot(sol, idxs = [:test])
display(pl)

dosing_test2 = [(t = 0.0, dose = 10, state = :d), (t = 0.5, dose = 20, state = :d), (t = 3.0, dose = 50, state = :d)]
test_person2 = EpilepsyModels.Person(dosing = dosing_test2)
pk_model2 = PKBasic()
sol2 = EpilepsyModels.solve_PK(pk_model2, pk_model2.θ, test_person2, endpoint = 5.0, options = ODE_options)
pl = plot(sol2, idxs = [:d, :s])
display(pl)

default_doses = (a = 2.0, b=3.0)
distr_first = (a = 1/6, b = 5/6)
distr_second = (a=5/6, b = 1/6)
prob_second = 1.0

dose_test = PolyDoses(default_doses, distr_first, distr_second, prob_second = prob_second)
dose_test2 = PolyDoses(default_doses, distr_first, distr_second, prob_second = prob_second, assign_not_supported = true)
person = EpilepsyModels.Person()
person2 = EpilepsyModels.Person()
EpilepsyModels.assign_dose!(dose_test, person, names = (d = (:b,),), wo_treatment = 2.0)
EpilepsyModels.assign_dose!(dose_test2, person2, names = (d= (:a,),), wo_treatment = 2.0)
=#
