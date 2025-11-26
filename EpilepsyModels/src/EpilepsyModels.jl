module EpilepsyModels

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Optimization
using ForwardDiff
using ComponentArrays

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

#For (de)transfering certain components in parameter vector into logscale
function partial_transform_to_logscale!(θ::ComponentArray; logscale::Tuple{String} = (), detransform::Bool = false)
    #set whether transform or detransform
    if detransform
        f = exp
    else 
        f = log
    end
    #search for matching labels, label2index returns vector of matching
    for label in labels(θ.PK)
        if label in logscale
            indices = label2index(θ.PK,label)
            for index in indices
                θ.PK[index] = f(θ.PK[index])
            end
        end
    end
    for label in labels(θ.Seizure)
        if label in logscale
            indices = label2index(θ.Seizure,label)
            for index in indices
                θ.Seizure[index] = f(θ.Seizure[index])
            end
        end
    end
end

#data should be [person structs], save seizure, measurement and dosing data in persons
#p contains m: model, data: tuple, logscale: Tuple{String}
#expects parameters in logscale tuple in logscale, internally detransforms in place
function get_negloglikelihood(θ::ComponentArray, p::NamedTuple) 
    #check if either model has random effects
    #if has_random_effects(m.pk_model) || has_random_effects(m.seizure_model)
        #do something to handle them
    #    return negloglikeli
    m = p.m
    data = p.data
    logscale = p.logscale
    options = p.options
    #for keys in logscale take exponential in θ
    partial_transform_to_logscale!(θ, logscale = logscale, detransform = true)
    loglikeli = zero(eltype(θ))
    for person in data
        sol = solve_PK(m.pk_model, θ.PK, person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time), options = options)
        if !(SciMLBase.successful_retcode(sol))
            return Inf
        end
        loglikeli += get_PK_loglikelihood(θ.PK, person; sol=sol)
        loglikeli += + get_seizure_loglikelihood(θ.Seizure, m.seizure_model, sol, person)
    end
    return -loglikeli
end

function optimise(m::FullModel, data::AbstractVector; maxiters::Int64 = 10^4, logscale::Tuple{String} = (), solver_optim = LBFGS(linesearch = LineSearches.BackTracking()), ODE_options = [AutoTsit5(Rosenbrock23())])
    #check if either model has random effects
    #if has_random_effects(m.pk_model) || has_random_effects(m.seizure_model)
        #do something to handle them
    negloglikeli = get_negloglikelihood
    θ_0 = ComponentArray((PK = m.pk_model.θ, Seizure = m.seizure_model.θ)) 
    #for keys in logscale transform to logscale in θ_0
    partial_transform_to_logscale!(θ_0, logscale = logscale)
    p = (m = m, data = data, logscale = logscale, options = ODE_options)
    objective = OptimizationFunction(negloglikeli, Optimization.AutoForwardDiff())
    problem = OptimizationProblem(objective, θ_0, p)
    estimate = solve(problem, solver_optim, maxiters = maxiters) 
    #transform parameters back into non logscale
    partial_transform_to_logscale!(estimate.u, logscale = logscale, detransform = true)
    print("Estimate: ", estimate)
    return estimate
end

#m determines model parts, n determines number of people, timepoints for measurements
function generate_data(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; timepoints::AbstractVector = 0:14.0:time, wo_treatment::AbstractFloat = 3.0, ODE_options = [AutoTsit5(Rosenbrock23())])
    population = generate_population(m.population_gen, n)
    for person in population
        assign_dose!(m.dose_gen, person, timeframe = time, wo_treatment = wo_treatment)
        sol = generate_measurements!(m.pk_model, person, timepoints = timepoints, endpoint = time, options = ODE_options)
        generate_seizures!(m.seizure_model, sol, person, start = 0.0, day_number = time)
        #note for time = 10 seizure counts end on day 9 (end on midnight between day 9 and 10)
    end
    return population
end

#for later when want to update doses etc regularly, update_reg better as int for seizure model
function generate_data_updating(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; update_reg::Int, timepoints::AbstractVector = 0:14.0:time, wo_treatment::AbstractFloat = 3.0, ODE_options = [AutoTsit5(Rosenbrock23())])
    population = generate_population(m.population_gen, n)
    for person in population
        passed_time = min(wo_treatment, time)
        #here generate for min(wo_treatment,time)
        assign_dose!(m.dose_gen, person, timeframe = passed_time, wo_treatment = wo_treatment)
        sol = generate_measurements!(m.pk_model, person, timepoints = timepoints, endpoint = passed_time, options = ODE_options)
        generate_seizures!(m.seizure_model, sol, person, start = 0.0, day_number = passed_time)
        while passed_time < time
            increment = max(time, passed_time + update_reg) - passed_time
            passed_time += increment
            current_timepoints = [t for t in timepoints if (passed_time-increment)<= t < passed_time] #filter timepoints in this interval
            assign_dose!(m.dose_gen, person, timeframe = increment)
            sol = generate_measurements!(m.pk_model, person, timepoints = current_timepoints, endpoint = passed_time, options = ODE_options)
            generate_seizures!(m.seizure_model, sol, person, start = (passed_time-increment), day_number = increment)
        end
    end
    return population
end

end # module EpilepsyModels
