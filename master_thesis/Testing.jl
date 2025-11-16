using Pkg

include("Full_model.jl")
using .EpilepsyModels
using Parameters
using ComponentArrays


println("Test")
Input_θ = (PK = ComponentArray((k_el = 2.0, k_abs = 5.0, σ=1.0)), 
            Seizure = ComponentArray((a = 6.0, b = 2.0)))
pk_model = PKBasic(θ=Input_θ.PK)
seizure_model = SeizureBasic(θ = Input_θ.Seizure)
mod = FullModel(pk_model, seizure_model, BasicPersonGenerator(), BasicDoses())
data = generate_data(mod, 20, 30.0, timepoints = 0:5.0:30)
println("Generated")
test_mod = FullModel(PKBasic(), SeizureBasic(), BasicPersonGenerator(), BasicDoses())
optimise(test_mod, data)
println("True θ:", Input_θ)
println("Done")