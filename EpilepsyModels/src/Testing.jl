using Pkg

#Pkg.develop(path = ".//EpilepsyModels")
include("EpilepsyModels.jl")

using .EpilepsyModels
using ComponentArrays
using OptimizationOptimJL
using LineSearches
using DifferentialEquations
using Plots

#Put on top to adjust: algorithm ODE solver
println("Test")
Input_θ = ComponentArray((PK = ComponentArray((k_el = 2.0, k_abs = 5.0, σ=0.2)), 
            Seizure = ComponentArray((a = 4, b = [-0.05]))))
Maxiters_optimiser = 1000
Population_size = 20
wo_treatment = 10.0
Obs_Duration = 40.0 + wo_treatment
PK_timepoints = wo_treatment:3.75:Obs_Duration
logscale = ("σ",)
solver_optim = LBFGS(linesearch = LineSearches.BackTracking())
ODE_options = [AutoTsit5(Rosenbrock23())]

pk_model = PKBasic(θ=Input_θ.PK)
seizure_model = SeizureBasic(θ = Input_θ.Seizure)
mod = FullModel(pk_model, seizure_model, BasicPersonGenerator(), BasicDoses())
data = generate_data(mod, Population_size, Obs_Duration, timepoints = PK_timepoints, wo_treatment = wo_treatment, ODE_options = ODE_options)
println("Generated")

test_mod = FullModel(PKBasic(), SeizureBasic(), BasicPersonGenerator(), BasicDoses())
estimate = optimise(test_mod, data, maxiters = Maxiters_optimiser, logscale = logscale, solver_optim = solver_optim, ODE_options = ODE_options)
#Transform partially to logscale for likelihood, gets detransformed in place in likelihood
EpilepsyModels.partial_transform_to_logscale!(Input_θ, logscale = logscale)
names = EpilepsyModels.get_keys_PK(mod.pk_model)
println("True Objective Value: ", EpilepsyModels.get_negloglikelihood(Input_θ, (m=mod, data=data, logscale=logscale, options=ODE_options, names=names)))
println("True θ:", Input_θ)

#Plot PK behavior (for each drug)
sol = EpilepsyModels.solve_PK(mod.pk_model, mod.pk_model.θ, data[1], endpoint = Obs_Duration)
for s in names.s
    pl = plot(sol, idxs = s, labels=["Concentration $(s)"], 
        xlabel="Time", ylabel="Amount", title="Basic PK Trajectory of $(s)")
    for i in eachindex(data)
        x_values = [measurement.timepoint for measurement in data[i].measurements if (measurement.state == s)]
        y_values = [measurement.measurement for measurement in data[i].measurements if (measurement.state == s)]
        plot!(x_values, y_values, seriestype = :scatter, mc = :green, label = "")
    end
    #add estimate plot
    Estimate_θ = estimate.u
    sol2 = EpilepsyModels.solve_PK(mod.pk_model, Estimate_θ.PK, data[1], endpoint = Obs_Duration)
    pl = plot!(sol2, idxs = s, labels=["Estimated concentration $(s)"], linecolor = :red)

    display(pl)
end

println("Done")