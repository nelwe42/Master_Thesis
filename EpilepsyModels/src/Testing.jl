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

println("Included")
#Input_θ = ComponentArray((PK = ComponentArray((k_el = 2.0, k_abs = 5.0, σ=0.2)), 
#           Seizure = ComponentArray((a = 4, b = [-0.05]))))
Input_θ = ComponentArray((PK = ComponentArray((k_abs = 8.0, c1 = 6.0, c2 = 0.25, c3 = 0.6, v1 = 50, v2 = 0.9, σ=0.1)), 
           Seizure = ComponentArray((a = 4, b = SA[-0.05]))))
Maxiters_optimiser = 1000
Population_size = 2 #20
wo_treatment = 3.0 #10.0
Obs_Duration = wo_treatment + 20.0 #+40.0
PK_timepoints = wo_treatment:3.75:Obs_Duration
logscale = ("σ",)
solver_optim = LBFGS(linesearch = LineSearches.BackTracking())
ODE_options = (AutoTsit5(Rosenbrock23()),)

pk_model = PKLEV(θ=Input_θ.PK)
seizure_model = SeizureBasic(θ = Input_θ.Seizure)
mod = FullModel(pk_model, seizure_model, PersonGeneratorLEV(), BasicDoses())
data = generate_data(mod, Population_size, Obs_Duration, timepoints = PK_timepoints, wo_treatment = wo_treatment, ODE_options = ODE_options)
println("Generated")

test_mod = FullModel(PKLEV(), SeizureBasic(), PersonGeneratorLEV(), BasicDoses())
estimate = optimise(test_mod, data, maxiters = Maxiters_optimiser, logscale = logscale, solver_optim = solver_optim, ODE_options = ODE_options)
#True values for comparison
println("True Objective Value: ", get_negloglikelihood_evaluated(Input_θ, mod, data, logscale = logscale, ODE_options = ODE_options))
println("True θ:", Input_θ)

#Plot PK behavior (for each drug)
names = EpilepsyModels.get_keys_PK(mod.pk_model)
i = 1 #index of person for which plotting is done
sol = EpilepsyModels.solve_PK(mod.pk_model, mod.pk_model.θ, data[i], endpoint = Obs_Duration)
covariates = NamedTuple{mod.pk_model.cov}(data[i].covariates)
sol2 = EpilepsyModels.solve_ODE(mod.pk_model, dosing = data[i].dosing, covariates = covariates, endpoint = Obs_Duration)
for s in names.s
    pl = plot(sol, idxs = s, labels=["Concentration $(s)"], 
        xlabel="Time", ylabel="Amount", title="PK Trajectory of $(s)")
    plot!(sol2, idxs = s, labels = ["Solve_ODE for $(s)"], linecolour = :green)
    x_values = [measurement.timepoint for measurement in data[i].measurements if (measurement.state[2] == s)]
    y_values = [measurement.measurement for measurement in data[i].measurements if (measurement.state[2] == s)]
    plot!(x_values, y_values, seriestype = :scatter, mc = :purple, label = "")
    #add estimate plot
    Estimate_θ = estimate.u
    sol2 = EpilepsyModels.solve_PK(mod.pk_model, Estimate_θ.PK, data[i], endpoint = Obs_Duration)
    pl = plot!(sol2, idxs = s, labels=["Estimated concentration $(s)"], linecolor = :red)

    display(pl)
end

println("Done")