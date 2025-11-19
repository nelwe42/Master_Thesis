using Pkg

include("Full_model.jl")
using .EpilepsyModels
using Parameters
using ComponentArrays
using Plots


println("Test")
Input_θ = (PK = ComponentArray((k_el = 2.0, k_abs = 5.0, σ=0.2)), 
            Seizure = ComponentArray((a = 4.0, b = -0.05)))
#For these values integrated drug exposure in one day roughly 2.5
#Likelihood value at Input_θ 1220.4461938648478
pk_model = PKBasic(θ=Input_θ.PK)
seizure_model = SeizureBasic(θ = Input_θ.Seizure)
mod = FullModel(pk_model, seizure_model, BasicPersonGenerator(), BasicDoses())
data = generate_data(mod, 20, 30.0, timepoints = 0:3.75:30)
println("Generated")

test_mod = FullModel(PKBasic(), SeizureBasic(), BasicPersonGenerator(), BasicDoses())
estimate = optimise(test_mod, data, maxiters = 10)
println("True θ:", Input_θ)

#Plot PK behavior
sol = EpilepsyModels.solve_PK(mod.pk_model, mod.pk_model.θ, data[1], endpoint = 30.0)
pl = plot(sol, idxs = (:s), labels=["Concentration (s)"], 
     xlabel="Time", ylabel="Amount", title="Basic PK Trajectory")
for i in eachindex(data)
    x_values = [measurement.timepoint for measurement in data[i].measurements]
    y_values = [measurement.measurement for measurement in data[i].measurements]
    plot!(x_values, y_values, seriestype = :scatter, label = "")
end
#add estimate plot
Estimate_θ = estimate.u
sol2 = EpilepsyModels.solve_PK(mod.pk_model, Estimate_θ.PK, data[1], endpoint = 30.0)
pl = plot!(sol2, idxs = (:s), labels=["Estimated concentration (s)"], linecolor = :red)

display(pl)

println("Done")

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Distributions
using DifferentialEquations

@mtkmodel Test begin
    @parameters begin
        k_el = 2.0
        k_abs = 5.0
        σ = 0.2
    end
    @variables begin
        d(t)[1:1] = [0.0]  # depot compartment - no drug at beginning
        s(t)[1:1] = [0.0]  # internal/central compartment
        S(t)[1:1] = [0.0]  #Integral over dose, always compute since don't know what seizure model requires
        obs(t)[1:1]
    end
    @equations begin
        D(d[1]) ~ -k_abs * d[1]
        D(s[1]) ~ k_abs * d[1] - k_el * s[1]
        D(S[1]) ~ s[1]
        obs[1] ~ Normal(s[1], σ)
    end
end

@mtkcompile ode_system = Test()

callback_set = EpilepsyModels.create_dosing_callbacks(data[1].dosing, ode_system)
problem = ODEProblem{true, SciMLBase.FullSpecialize}(ode_system, [], (0.0, 30.0), callback = callback_set)
sol = solve(problem, Tsit5())