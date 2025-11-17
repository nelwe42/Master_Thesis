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
#Likelihood value at optimum 1233.2663851517918
pk_model = PKBasic(θ=Input_θ.PK)
seizure_model = SeizureBasic(θ = Input_θ.Seizure)
mod = FullModel(pk_model, seizure_model, BasicPersonGenerator(), BasicDoses())
data = generate_data(mod, 20, 30.0, timepoints = 0:3.75:30)
println("Generated")

#Plot PK behavior
sol = EpilepsyModels.solve_PK(mod.pk_model, mod.pk_model.θ, data[1], endpoint = 30.0)
pl = plot(sol, idxs = (:s), labels=["Concentration (s)"], 
     xlabel="Time", ylabel="Amount", title="Basic PK Trajectory")
for i in eachindex(data)
    x_values = [measurement.timepoint for measurement in data[i].measurements]
    y_values = [measurement.measurement for measurement in data[i].measurements]
    plot!(x_values, y_values, seriestype = :scatter, label = "")
end
display(pl)

test_mod = FullModel(PKBasic(), SeizureBasic(), BasicPersonGenerator(), BasicDoses())
#optimise(test_mod, data)
#println("True θ:", Input_θ)
println("Done")