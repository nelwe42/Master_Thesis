module EpilepsyModels

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Optimization
using ForwardDiff
using ComponentArrays

export optimise, generate_data, generate_data_updating, BasicDoses, PKBasic, PKLEV, BasicPersonGenerator, 
SeizureBasic, FullModel, PersonGeneratorLEV

include("Person Generator.jl")
include("Dose Generator.jl")
include("PK Model.jl")
include("Seizure Model.jl")

struct FullModel{PK<:PKModel, S<:SeizureModel, P<:PersonGenerator, D<:DoseGenerator}
    pk_model::PK
    seizure_model::S
    population_gen::P
    dose_gen::D
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
                @inbounds θ.PK[index] = f(θ.PK[index])
            end
        end
    end
    for label in labels(θ.Seizure)
        if label in logscale
            indices = label2index(θ.Seizure,label)
            for index in indices
                @inbounds θ.Seizure[index] = f(θ.Seizure[index])
            end
        end
    end
end

#data should be (person structs), save seizure, measurement and dosing data in persons
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
    names = p.names
    problems = p.problems
    #for keys in logscale take exponential in θ
    partial_transform_to_logscale!(θ, logscale = logscale, detransform = true)
    loglikeli = zero(eltype(θ))
    for i in eachindex(data)
        @inbounds sol = solve_PK(problems[i], θ.PK, options = options)
        if !(SciMLBase.successful_retcode(sol))
            return Inf
        end
        @inbounds loglikeli += get_PK_loglikelihood(θ.PK, data[i], sol=sol)
        @inbounds loglikeli += get_seizure_loglikelihood(θ.Seizure, m.seizure_model, sol, data[i], names=names)
    end
    return -loglikeli
end

function optimise(m::FullModel, data::Tuple; maxiters::Int64 = 10^4, logscale::Tuple{String} = (), solver_optim = LBFGS(linesearch = LineSearches.BackTracking()), ODE_options = (AutoTsit5(Rosenbrock23())))
    #check if either model has random effects
    #if has_random_effects(m.pk_model) || has_random_effects(m.seizure_model)
        #do something to handle them
    names = get_keys_PK(m.pk_model)
    negloglikeli = get_negloglikelihood
    #create ODE problem for each person in data
    sys = create_ode_system(m)
    problems = (create_problem(m, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time)) for person in data)
    #create initial guess
    θ_0 = ComponentArray((PK = m.pk_model.θ, Seizure = m.seizure_model.θ)) 
    #for keys in logscale transform to logscale in θ_0
    partial_transform_to_logscale!(θ_0, logscale = logscale)
    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems)
    objective = OptimizationFunction(negloglikeli, Optimization.AutoForwardDiff())
    problem = OptimizationProblem(objective, θ_0, p)
    estimate = solve(problem, solver_optim, maxiters = maxiters) 
    #transform parameters back into non logscale
    partial_transform_to_logscale!(estimate.u, logscale = logscale, detransform = true)
    print("Estimate: ", estimate)
    return estimate
end

#m determines model parts, n determines number of people, timepoints for measurements
function generate_data(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; timepoints::AbstractVector = 0:14.0:time, wo_treatment::AbstractFloat = 3.0, ODE_options = (AutoTsit5(Rosenbrock23())))
    population = generate_population(m.population_gen, n)
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    for person in population
        assign_dose!(m.dose_gen, person, names= names, timeframe = time, wo_treatment = wo_treatment)
        sol = generate_measurements!(m.pk_model, sys, person, timepoints = timepoints, endpoint = time, options = ODE_options)
        generate_seizures!(m.seizure_model, sol, person, start = 0.0, day_number = time, names=names)
        #note for time = 10 seizure counts end on day 9 (end on midnight between day 9 and 10)
    end
    return population
end

#for later when want to update doses etc regularly
function generate_data_updating(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; update_reg::AbstractFloat = time, timepoints::AbstractVector = 0:14.0:time, wo_treatment::AbstractFloat = 3.0, ODE_options = [AutoTsit5(Rosenbrock23())])
    population = generate_population(m.population_gen, n)
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    for person in population
        passed_time = min(wo_treatment, time)
        #here generate for min(wo_treatment,time)
        assign_dose!(m.dose_gen, person, names=names, timeframe = passed_time, wo_treatment = wo_treatment)
        sol = generate_measurements!(m.pk_model, sys, person, timepoints = timepoints, endpoint = passed_time, options = ODE_options)
        generate_seizures!(m.seizure_model, sol, person, start = 0.0, day_number = floor(passed_time), names=names)
        seizure_time_rest = passed_time - floor(passed_time)
        while passed_time < time
            increment = max(time, passed_time + update_reg) - passed_time
            passed_time += increment
            current_timepoints = [t for t in timepoints if (passed_time-increment)<= t < passed_time] #filter timepoints in this interval
            assign_dose!(m.dose_gen, person, names=names, timeframe = increment)
            #when later in generate_measurements do solve from make sure to adjust start with seizure_time_rest here
            sol = generate_measurements!(m.pk_model, sys, person, timepoints = current_timepoints, endpoint = passed_time, options = ODE_options)
            generate_seizures!(m.seizure_model, sol, person, start = (passed_time-increment-seizure_time_rest), day_number = floor(increment+seizure_time_rest), names=names)
            seizure_time_rest = increment + seizure_time_rest - floor(increment + seizure_time_rest)
        end
    end
    return population
end

end # module EpilepsyModels
