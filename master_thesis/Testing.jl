using Pkg

include("Full_model.jl")
using .EpilepsyModels
using Parameters
using ComponentArrays


println("Test")
pk_model = PKBasic(θ=ComponentArray((k_el = 2.0, k_abs = 5.0, σ=1.0)))
seizure_model = SeizureBasic(θ=ComponentArray((a = 6.0, b = 2.0)))
mod = FullModel(pk_model, seizure_model, BasicPersonGenerator(), BasicDoses())
data = generate_data(mod, 20, 30.0, timepoints = 0:5.0:30)
println("Generated")
optimise(mod, data)
println("Done")