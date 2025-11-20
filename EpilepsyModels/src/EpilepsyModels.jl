module EpilepsyModels

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Optimization
using ForwardDiff
using ComponentArrays
using OptimizationOptimJL
using LineSearches

export optimise, generate_data, generate_data_updating, BasicDoses, PKBasic, BasicPersonGenerator, 
SeizureBasic, FullModel

include("Person Generator.jl")
include("Dose Generator.jl")
include("PK Model.jl")
include("Seizure Model.jl")

struct FullModel
    pk_model::PKModel
    seizure_model::SeizureModel
    population_gen::PersonGenerator
    dose_gen::DoseGenerator
end


#data should be [person structs], save seizure, measurement and dosing data in persons
#p contains m: model and data: tuple
function get_negloglikelihood(θ::ComponentArray, p::NamedTuple) 
    #check if either model has random effects
    #if has_random_effects(m.pk_model) || has_random_effects(m.seizure_model)
        #do something to handle them
    #    return negloglikeli
    m = p.m
    data = p.data
    loglikeli = zero(eltype(θ))
    for i in eachindex(data)
        person = data[i]
        sol = solve_PK(m.pk_model, θ.PK, person, endpoint = person.measurements[end].timepoint)
        if !(SciMLBase.successful_retcode(sol))
            return Inf
        end
        loglikeli = loglikeli + get_PK_loglikelihood(θ.PK, person; sol=sol)
        loglikeli = loglikeli + get_seizure_loglikelihood(θ.Seizure, m.seizure_model, sol, person)
    end
    return -loglikeli
end

function optimise(m::FullModel, data::AbstractVector; maxiters::Int64 = 10^4)
    #check if either model has random effects
    #if has_random_effects(m.pk_model) || has_random_effects(m.seizure_model)
        #do something to handle them
    negloglikeli = get_negloglikelihood
    θ_0 = ComponentArray((PK = m.pk_model.θ, Seizure = m.seizure_model.θ)) 
    p = (m = m, data = data)
    objective = OptimizationFunction(negloglikeli, Optimization.AutoForwardDiff())
    problem = OptimizationProblem(objective, θ_0, p)
    estimate = solve(problem, LBFGS(linesearch = LineSearches.BackTracking()), maxiters = maxiters) 
    println("Estimate:", estimate)
    return estimate
end

#m determines model parts, n determines number of people, timepoints for measurements
function generate_data(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; timepoints::AbstractVector = 0:14.0:time)
    population = generate_population(m.population_gen, n)
    for i in eachindex(population)
        person = population[i]
        assign_dose!(m.dose_gen, person, timeframe = time)
        sol = generate_measurements!(m.pk_model, person, timepoints = timepoints)
        generate_seizures!(m.seizure_model, sol, person, start = 0.0, day_number = time)
        #note for time = 10 seizure counts end on day 9 (end on midnight between day 9 and 10)
    end
    return population
end

#for later when want to update doses etc regularly, update_reg better as int for seizure model
function generate_data_updating(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; update_reg::Int, timepoints::AbstractVector = 0:14.0:time)
    population = generate_population(m.population_gen, n)
    for i in eachindex(population)
        person = population[i]
        passed_time = 0
        while passed_time < time
            increment = max(time, passed_time + update_reg) - passed_time
            passed_time = passed_time + increment
            current_timepoints = [t for t in timepoints if (passed_time-increment)<= t < passed_time] #filter timepoints in this interval
            assign_dose!(m.dose_gen, person, timeframe = increment)
            sol = generate_measurements!(m.pk_model, person, timepoints = current_timepoints)
            generate_seizures!(m.seizure_model, sol, person, start = (passed_time-increment), day_number = increment)
        end
    end
end

end # module EpilepsyModels
