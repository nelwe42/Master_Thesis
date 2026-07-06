module EpilepsyModels

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Optimization
using ForwardDiff
using ComponentArrays
using FiniteDiff
using Parameters
using LinearAlgebra
using OptimizationOptimJL
using LogDensityProblems
using LogDensityProblemsAD
using AdvancedHMC
using AdvancedMH
using MCMCChains

export optimise, optimise_hierarchical, optimise_sampled, generate_data, generate_data_updating, generate_data_modified, get_negloglikelihood_evaluated, get_negloglikelihood_evaluated_hierarchical, plot_fit, multi_data_run,
BasicDoses, PolyDosesRandom, PolyDoses, BigFourDoses, PKBasic, PKLEV, PKLEVNoAbsorption, PKCBZ, PKVPA, PKLTG, PKBigFour,
BasicPersonGenerator, PersonGeneratorLEV, BigFourPersonGenerator, SeizureBasic, SeizureNegativeBinomial, SeizureVPA, SeizureMult, FullModel

include("Person Generator.jl")
include("PK Model.jl")
include("Dose Generator.jl")
include("Seizure Model.jl")

@with_kw struct FullModel{PK<:PKModel, S<:SeizureModel, P<:PersonGenerator, D<:DoseGenerator}
    pk_model::PK
    seizure_model::S
    population_gen::P = BasicPersonGenerator()
    dose_gen::D = BasicDoses()
end

#For (de)transfering certain components in parameter vector into logscale
function partial_transform_to_logscale!(θ::ComponentArray; logscale::Tuple{Vararg{String}} = (), detransform::Bool = false)
    #search for matching labels, label2index returns vector of matching
    for label in logscale
        #To handle both transforming whole vector valued parameter or individual indices
        if label in labels(θ.PK) || Symbol(label) in keys(θ.PK)
            indices = label2index(θ.PK,label)
            for index in indices
                if detransform
                    @inbounds θ.PK[index] = exp(θ.PK[index])
                else
                    @inbounds θ.PK[index] = log(max(θ.PK[index], eps(eltype(θ))))
                end
            end
        end

        if label in labels(θ.Seizure) || Symbol(label) in keys(θ.Seizure)
            indices = label2index(θ.Seizure,label)
            for index in indices
                if detransform
                    @inbounds θ.Seizure[index] = exp(θ.Seizure[index])
                else
                    @inbounds θ.Seizure[index] = log(max(θ.Seizure[index], eps(eltype(θ))))
                end
            end
        end
    end
end

#For (de)transfering certain components in parameter vector into logscale for only one model part (Seizure or PK)
function partial_transform_to_logscale_partwise!(θ::ComponentArray; logscale::Tuple{Vararg{String}} = (), detransform::Bool = false)
    #search for matching labels, label2index returns vector of matching
    for label in logscale
        #To handle both transforming whole vector valued parameter or individual indices
        if label in labels(θ) || Symbol(label) in keys(θ)
            indices = label2index(θ,label)
            for index in indices
                if detransform
                    @inbounds θ[index] = exp(θ[index])
                else
                    @inbounds θ[index] = log(max(θ[index], eps(eltype(θ))))
                end
            end
        end
    end
end

#data should be (person structs), save seizure, measurement and dosing data in persons
#p tuple of general information about model, data etc, should be created in function calling this one, e.g. get_negloglikelihood_evaluated
#expects parameters in logscale tuple in logscale, internally detransforms 
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
    system = p.system
    indices = p.indices_θ
    if hasproperty(p, :max_threads)
        thread_num = max(1, p.max_threads)
    else
        thread_num = 1
    end
    individuals = length(data)
    people_per_thread = Int(ceil(individuals/thread_num))
    θ_use = copy(θ)
    #for keys in logscale take exponential in θ
    partial_transform_to_logscale!(θ_use, logscale = logscale, detransform = true)
    loglikeli = zero(eltype(θ_use))
    try
        loglikelihoods = Array{Union{Float64, ForwardDiff.Dual}}(undef, individuals)
        #If need all solutions at once for partial likelihood, collect them here
        if p.cont_seizure
            sols = Array{SciMLBase.AbstractNoTimeSolution}(undef, individuals)
        end
        keep_going = Threads.Atomic{Bool}(true)
        Threads.@threads for j in 1:min(individuals, thread_num)
            for i in ((j-1)*people_per_thread+1):min(j*people_per_thread, individuals)
                if !keep_going[]
                    break
                end
                @inbounds sol = solve_PK(problems[i], system, θ_use.PK, indices_θ = indices, options = options)
                if !(SciMLBase.successful_retcode(sol))
                    keep_going[] = false
                    break
                end
                @inbounds loglikelihoods[i] = get_PK_loglikelihood(θ_use.PK, data[i], sol=sol)
                #Check if can evaluate seizure likelihood now or need all data for partial likelihood
                if p.cont_seizure
                    sols[i] = sol
                else
                    @inbounds loglikelihoods[i] += get_seizure_loglikelihood(θ_use.Seizure, m.seizure_model, sol, data[i], names=names)
                end
            end
        end
        if !keep_going[]
            return Inf
        end
        if p.cont_seizure
            loglikeli = sum(loglikelihoods)
            loglikeli += get_seizure_loglikelihood(θ_use.Seizure, m.seizure_model, sols, data, names=names)
            return -loglikeli
        else
            return -sum(loglikelihoods)
        end
    catch e
        fail_hard = hasproperty(p, :objective_fail_hard) ? p.objective_fail_hard : false
        if fail_hard
            rethrow(e)
        end
        if hasproperty(p, :objective_warned_ref) && hasproperty(p, :objective_warn)
            if p.objective_warn && !p.objective_warned_ref[]
                p.objective_warned_ref[] = true
                @warn "Objective evaluation failed; returning Inf for this candidate." exception = e #exception=(e, catch_backtrace())
            end
        end
        return Inf
    end
end

function get_negloglikelihood_PK(θ::ComponentArray, p::NamedTuple) 
    data = p.data
    logscale = p.logscale
    options = p.options
    problems = p.problems
    system = p.system
    indices = p.indices_θ
    θ_use = copy(θ)
    #for keys in logscale take exponential in θ
    partial_transform_to_logscale_partwise!(θ_use, logscale = logscale, detransform = true)
    loglikeli = zero(eltype(θ_use))
    try
        for i in eachindex(data)
            @inbounds sol = solve_PK(problems[i], system, θ_use, indices_θ = indices, options = options)
            if !(SciMLBase.successful_retcode(sol))
                return Inf
            end
            @inbounds loglikeli += get_PK_loglikelihood(θ_use, data[i], sol=sol)
        end
        return -loglikeli
    catch e
        fail_hard = hasproperty(p, :objective_fail_hard) ? p.objective_fail_hard : false
        if fail_hard
            rethrow(e)
        end
        if hasproperty(p, :objective_warned_ref) && hasproperty(p, :objective_warn)
            if p.objective_warn && !p.objective_warned_ref[]
                p.objective_warned_ref[] = true
                @warn "Objective evaluation failed; returning Inf for this candidate." exception = e #exception=(e, catch_backtrace())
            end
        end
        return Inf
    end
end

function get_negloglikelihood_Seizure(θ::ComponentArray, p::NamedTuple) 
    m = p.m
    data = p.data
    logscale = p.logscale
    names = p.names
    solutions = p.solutions
    θ_use = copy(θ)
    #for keys in logscale take exponential in θ
    partial_transform_to_logscale_partwise!(θ_use, logscale = logscale, detransform = true)
    loglikeli = zero(eltype(θ_use))
    if p.cont.seizure
        loglikeli = get_seizure_loglikelihood(θ_use, m.seizure_model, solutions, data, names=names)
    else
        for i in eachindex(data)
            sol = solutions[i]
            @inbounds loglikeli += get_seizure_loglikelihood(θ_use, m.seizure_model, sol, data[i], names=names)
        end
    end
    return -loglikeli
end

#some optimisers cannot work with componentarrays directly
function get_negloglikelihood_vectorised(θ::AbstractVector, p::NamedTuple)
    θ_struct = ComponentArray(copy(θ), p.axes_θ)
    return get_negloglikelihood(θ_struct, p)
end

function get_negloglikelihood_evaluated(θ::ComponentArray, m::FullModel, data::Tuple; logscale::Tuple{Vararg{String}} = (), ODE_options = (AutoTsit5(Rosenbrock23()),))
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    problems = Tuple(create_problem(m.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time[2])) for person in data)
    indices_θ = [ModelingToolkit.parameter_index(sys, x).idx for x in keys(θ.PK)]
    θ_use = deepcopy(θ)
    partial_transform_to_logscale!(θ_use, logscale = logscale)
    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ, cont_seizure = (m.seizure_model isa SeizureModelContinuous))
    negloglikeli = get_negloglikelihood(θ_use, p)
    return negloglikeli
end

function get_negloglikelihood_evaluated_hierarchical(θ::ComponentArray, m::FullModel, data::Tuple; logscale::Tuple{Vararg{String}} = (), ODE_options = (AutoTsit5(Rosenbrock23()),))
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    problems = Tuple(create_problem(m.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time[2])) for person in data)
    indices_θ = [ModelingToolkit.parameter_index(sys, x).idx for x in keys(θ.PK)]
    θ_use = deepcopy(θ)
    #Solve before PK gets transferred into logscale
    solutions = [solve_PK(problems[i], sys, θ_use.PK, indices_θ = indices_θ, options = ODE_options) for i in eachindex(data)]
    if any(.!(SciMLBase.successful_retcode.(solutions)))
        error("Unsuccessful solve for given PK parameters")
    end
    partial_transform_to_logscale!(θ_use, logscale = logscale)
    p_PK = (data = data, logscale = logscale, options = ODE_options, problems = problems, system = sys, indices_θ = indices_θ, axes_θ = getaxes(θ_use))
    p_Seizure = (m = m, data = data, logscale = logscale, names = names, solutions = solutions)
    
    negloglikeli = (PK = get_negloglikelihood_PK(θ_use.PK, p_PK), Seizure = get_negloglikelihood_Seizure(θ_use.Seizure, p_Seizure))
    return negloglikeli
end

#generate multistart points randomly
function latin_hypercube_samples(n::Int, lower::AbstractVector, upper::AbstractVector; rng = Random.default_rng())
    d = length(lower)
    if length(upper) != d
        error("latin_hypercube_samples: lower and upper must have the same length")
    end
    if n < 1
        error("latin_hypercube_samples: n must be >= 1")
    end
    X = Matrix{Float64}(undef, n, d)
    for j in 1:d
        perm = Random.randperm(rng, n)
        width = upper[j] - lower[j]
        if !(isfinite(width) && width > 0)
            error("latin_hypercube_samples: bounds must satisfy upper > lower and be finite")
        end
        for i in 1:n
            u = (perm[i] - Random.rand(rng)) / n
            X[i, j] = lower[j] + u * width
        end
    end
    return X
end

function optimise_hierarchical(m::FullModel, data::Tuple; maxiters::Int64 = 10^4, logscale::Tuple{Vararg{String}} = (), run_CI::Bool = false, bound_abs::Union{Nothing, AbstractFloat} = nothing, lower_upper::Union{Nothing, Tuple{ComponentArray, ComponentArray}} = nothing, 
                objective_fail_hard::Bool = false, objective_warn::Bool = true, store_trace::Bool = false, printing::Bool = true, confidence::AbstractFloat = 0.95, finite_not_forward::Bool = false, sandwich::Bool = true, solver_optim = LBFGS(linesearch = LineSearches.BackTracking()), solver_options = (), ODE_options = (AutoTsit5(Rosenbrock23()),))
    
    #check if either model has random effects
    #if has_random_effects(m.pk_model) || has_random_effects(m.seizure_model)
        #do something to handle them
    names = get_keys_PK(m.pk_model)
    #create ODE problem for each person in data
    sys = create_ode_system(m.pk_model)
    problems = Tuple(create_problem(m.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time[2])) for person in data)
    #create initial guess
    θ_0 = ComponentArray((PK = m.pk_model.θ, Seizure = m.seizure_model.θ)) 
    #get indices for setting θ
    indices_θ = [ModelingToolkit.parameter_index(sys, x).idx for x in keys(θ_0.PK)]
    #for keys in logscale transform to logscale in θ_0
    θ_0_PK = m.pk_model.θ
    θ_0_Seizure = m.seizure_model.θ
    partial_transform_to_logscale_partwise!(θ_0_PK, logscale = logscale)
    partial_transform_to_logscale_partwise!(θ_0_Seizure, logscale = logscale)
    d = length(θ_0)

    #set bounds if required, handle if both individual and absolute bounds
    if !isnothing(lower_upper)
        lb, ub = lower_upper
        if length(ub) != d || length(lb) != d
            error("Upper and lower bounds must match parameter dimension $d")
        end
        if !isnothing(bound_abs)
            ub .=  min.(ub, bound_abs)
            if any(ub .< -bound_abs)
                error("Upper bounds too low to fulfill absolute bounds")
            end
            lb .=  max.(lb, -bound_abs)
            if any(lb .> bound_abs)
                error("Lower bounds too high to fulfill absolute bounds")
            end
        end
        #Check if bounds are valid
        if any(ub .< lb)
            error("Upper bounds strictly smaller than lower ones")
        end
    else
        if !isnothing(bound_abs)
            ub = ComponentArray([bound_abs for i in eachindex(θ_0)], getaxes(θ_0))
            lb = ComponentArray([-bound_abs for i in eachindex(θ_0)], getaxes(θ_0))
        else
            ub = nothing
            lb = nothing
        end
    end
    #ensure initial guess satisfies bounds
    if !isnothing(lb)
        θ_0 .= clamp.(θ_0, lb, ub)
    end

    #First fit PK model
    p_PK = (data = data, logscale = logscale, options = ODE_options, problems = problems, system = sys, indices_θ = indices_θ, axes_θ = getaxes(θ_0), objective_fail_hard = objective_fail_hard, objective_warn = objective_warn, objective_warned_ref = Ref(false))
    objective_PK = OptimizationFunction(get_negloglikelihood_PK, Optimization.AutoForwardDiff())
    if isnothing(lb)
        problem_PK = OptimizationProblem(objective_PK, θ_0_PK, p_PK)
    else 
        problem_PK = OptimizationProblem(objective_PK, θ_0_PK, p_PK, lb = lb.PK, ub = ub.PK)
    end
    estimate_PK = solve(problem_PK, solver_optim, maxiters = maxiters, store_trace = store_trace, solver_options...) 
    #transform parameters back into non logscale
    partial_transform_to_logscale_partwise!(estimate_PK.u, logscale = logscale, detransform = true)

    #Check solve successful
    solutions = [solve_PK(problems[i], sys, estimate_PK.u, indices_θ = indices_θ, options = ODE_options) for i in eachindex(data)]
    if !SciMLBase.successful_retcode(estimate_PK.retcode) || any(.!(SciMLBase.successful_retcode.(solutions)))
        error("Unsuccessful solve in PK estimation")
    end

    #Fit Seizure model
    p_Seizure = (m = m, data = data, logscale = logscale, names = names, solutions = solutions)
    objective_Seizure = OptimizationFunction(get_negloglikelihood_Seizure, Optimization.AutoForwardDiff())
    if isnothing(lb)
        problem_Seizure = OptimizationProblem(objective_Seizure, θ_0_Seizure, p_Seizure)
    else 
        problem_Seizure = OptimizationProblem(objective_Seizure, θ_0_Seizure, p_Seizure, lb = lb.Seizure, ub = ub.Seizure)
    end
    estimate_Seizure = solve(problem_Seizure, solver_optim, maxiters = maxiters, store_trace = store_trace, solver_options...) 
    #transform parameters back into non logscale
    partial_transform_to_logscale_partwise!(estimate_Seizure.u, logscale = logscale, detransform = true)

    if run_CI
        CI = inverse_hessian(estimate.u, p, confidence = confidence, logscale = logscale, finite_not_forward = finite_not_forward, sandwich = sandwich)
        estimate = (u = ComponentVector(PK = estimate_PK.u, Seizure = estimate_Seizure.u), retcode = estimate_Seizure.retcode, objective = (PK = estimate_PK.objective, Seizure = estimate_Seizure.objective), estimate_PK = estimate_PK, estimate_Seizure = estimate_Seizure, CI = CI)
    else
        estimate = (u = ComponentVector(PK = estimate_PK.u, Seizure = estimate_Seizure.u), retcode = estimate_Seizure.retcode, objective = (PK = estimate_PK.objective, Seizure = estimate_Seizure.objective), estimate_PK = estimate_PK, estimate_Seizure = estimate_Seizure)
    end
    if printing
        println("Estimate: ", estimate.u)
        println(estimate.retcode)
        println(estimate.objective)
    end
    return estimate
end

function optimise(m::FullModel, data::Tuple; maxiters::Int64 = 10^4, maxtime::AbstractFloat = Inf, logscale::Tuple{Vararg{String}} = (), run_CI::Bool = false, bound_abs::Union{Nothing, AbstractFloat} = nothing, lower_upper::Union{Nothing, Tuple{ComponentArray, ComponentArray}} = nothing,
    objective_fail_hard::Bool = false, objective_warn::Bool = true, store_trace::Bool = false, multistart::Int = 1, max_threads::Int = multistart, multistart_seed::Union{Nothing, Int} = nothing, multistart_include_initial::Bool = true, multistart_bounds::Union{Nothing, Tuple{AbstractVector, AbstractVector}, AbstractFloat} = nothing, 
    use_model_bounds::Bool = true, prefilter::Union{Int, Nothing} = nothing, custom_starts::Union{AbstractVector,Nothing} = nothing, noise_params::Tuple{Vararg{Tuple{Int, AbstractFloat}}} = (), printing::Bool = true, confidence::AbstractFloat = 0.95, finite_not_forward::Bool = false, sandwich::Bool = true,
    solver_optim = LBFGS(linesearch = LineSearches.BackTracking()), solver_options = (), ODE_options = (AutoTsit5(Rosenbrock23()),))
    
    #check if either model has random effects
    #if has_random_effects(m.pk_model) || has_random_effects(m.seizure_model)
        #do something to handle them
    names = get_keys_PK(m.pk_model)
    #create ODE problem for each person in data
    sys = create_ode_system(m.pk_model)
    problems = Tuple(create_problem(m.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time[2])) for person in data)
    #create initial guess
    θ_0 = ComponentArray((PK = m.pk_model.θ, Seizure = m.seizure_model.θ)) 
    #get indices for setting θ
    indices_θ = [ModelingToolkit.parameter_index(sys, x).idx for x in keys(θ_0.PK)]
    #for keys in logscale transform to logscale in θ_0
    partial_transform_to_logscale!(θ_0, logscale = logscale)
    
    axes_θ = getaxes(θ_0)
    θ_0_vec = collect(θ_0)
    d = length(θ_0_vec)

    #set bounds if required, handle if both individual and absolute bounds
    if !isnothing(lower_upper)
        lb, ub = lower_upper
        if length(ub) != d || length(lb) != d
            error("Upper and lower bounds must match parameter dimension $d")
        end
        if !isnothing(bound_abs)
            ub .=  min.(ub, bound_abs)
            if any(ub .< -bound_abs)
                error("Upper bounds too low to fulfill absolute bounds")
            end
            lb .=  max.(lb, -bound_abs)
            if any(lb .> bound_abs)
                error("Lower bounds too high to fulfill absolute bounds")
            end
        end
        #Check if bounds are valid
        if any(ub .< lb)
            error("Upper bounds strictly smaller than lower ones")
        end
    else
        if !isnothing(bound_abs)
            ub = ComponentArray([bound_abs for i in eachindex(θ_0)], getaxes(θ_0))
            lb = ComponentArray([-bound_abs for i in eachindex(θ_0)], getaxes(θ_0))
        else
            ub = nothing
            lb = nothing
        end
    end
    #Check if want to use model bounds and if they are defined
    if use_model_bounds
        if hasproperty(m.pk_model, :bounds) || hasproperty(m.seizure_model, :bounds)
            if !hasproperty(m.pk_model, :bounds)
                lb_model = ComponentArray(PK = ComponentArray([-Inf for e in m.pk_model.θ], getaxes(m.pk_model.θ)), Seizure = m.seizure_model.bounds.lb)
                ub_model = ComponentArray(PK = ComponentArray([Inf for e in m.pk_model.θ], getaxes(m.pk_model.θ)), Seizure = m.seizure_model.bounds.ub)
            elseif !hasproperty(m.seizure_model, :bounds)
                lb_model = ComponentArray(PK = m.pk_model.bounds.lb, Seizure = ComponentArray([-Inf for e in m.seizure_model.θ], getaxes(m.seizure_model.θ)))
                ub_model = ComponentArray(PK = m.pk_model.bounds.ub, Seizure = ComponentArray([Inf for e in m.seizure_model.θ], getaxes(m.seizure_model.θ)))
            else
                lb_model = ComponentArray(PK = m.pk_model.bounds.lb, Seizure = m.seizure_model.bounds.lb)
                ub_model = ComponentArray(PK = m.pk_model.bounds.ub, Seizure = m.seizure_model.bounds.ub)
            end
            #transform bounds to logscale where necessary
            partial_transform_to_logscale!(lb_model, logscale = logscale)
            partial_transform_to_logscale!(ub_model, logscale = logscale)
            #Check internal consistency and with lb, ub if defined
            if length(ub_model) != d || length(lb_model) != d
                error("Upper and lower model bounds must match parameter dimension $d")
            end
            if any(ub_model .< lb_model)
                error("Upper Model bounds strictly smaller than lower ones")
            end
            if !isnothing(lb)
                ub .=  min.(ub, ub_model)
                lb .=  max.(lb, lb_model)
                if any(lb .> ub)
                    error("Model and manual bounds cannot be satisfied at the same time")
                end
            else
                lb = lb_model
                ub = ub_model
            end
        else
            lb_model = nothing
            ub_model = nothing
        end
    end

    #Ensure initial guess satifies bounds
    if !isnothing(lb)
        θ_0 .= clamp.(θ_0, lb, ub)
    end

    #Check if can use ComponentArrays or solver requires normal Array
    if parentmodule(typeof(solver_optim)) in [Optim] #add packages here where ComponentArray works
        vectorised = false
    else
        vectorised = true
    end

    n_starts = max(multistart, 1)
    thread_num = max(max_threads, 1)
    internal_threads = Int(floor(thread_num/min(n_starts, thread_num)))
    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ, cont_seizure = (m.seizure_model isa SeizureModelContinuous), axes_θ = axes_θ, max_threads = internal_threads, objective_fail_hard = objective_fail_hard, objective_warn = objective_warn, objective_warned_ref = Ref(false))
    if vectorised
        objective = OptimizationFunction(get_negloglikelihood_vectorised, Optimization.AutoForwardDiff())
        #Change everything to vectors
        ub = collect(ub)
        lb = collect(lb)
        θ_0 = θ_0_vec
    else
        objective = OptimizationFunction(get_negloglikelihood, Optimization.AutoForwardDiff())
    end

    lower = zeros(Float64, d)
    upper = zeros(Float64, d)
    if !isnothing(multistart_bounds)
        if typeof(multistart_bounds) <: AbstractFloat
            lower .= Float64(-multistart_bounds)
            upper .= Float64(multistart_bounds)
        else
            lower_raw, upper_raw = multistart_bounds
            if length(lower_raw) != d || length(upper_raw) != d
                error("multistart_bounds must match parameter dimension $d")
            end
            lower .= Float64.(lower_raw)
            upper .= Float64.(upper_raw)
        end
    #check if lb, ub are defined and finite, try those instead
    elseif !isnothing(lb) && all(isfinite.(lb)) && all(isfinite.(ub))
        lower = lb
        upper = ub
    else
        #Fallback finite box around initial point in unconstrained mode.
        lower .= Float64.(θ_0_vec) .- 2.0
        upper .= Float64.(θ_0_vec) .+ 2.0
    end

    #ensure generated starts will satisfy bounds
    if !isnothing(lb)
        lower .= max.(lower, lb)
        upper .= min.(upper, ub)
    end

    if any(.!isfinite.(lower)) || any(.!isfinite.(upper)) || any(upper .<= lower)
        error("Invalid multistart bounds: require finite values and upper > lower component-wise")
    end

    if !isnothing(prefilter)
        if prefilter <= n_starts
            error("Amount of starts for prefiltering must be strictly greater than multistarts")
        else
            n_starts = prefilter
        end
    end
    starts = Matrix{Float64}(undef, n_starts, d)
    row_idx = 1
    if multistart_include_initial
        starts[row_idx, :] .= Float64.(θ_0_vec)
        row_idx += 1
    end
    n_lhs = n_starts - (multistart_include_initial ? 1 : 0)

    #Check if passed starts work
    if !isnothing(custom_starts) && !isempty(custom_starts)
        if any([length(start) != d for start in custom_starts])
            error("Passed custom starts do not match parameter dimension")
        end
        if length(custom_starts) > max(multistart,1) - (multistart_include_initial ? 1 : 0)
            error("Too many starts passed for multistart")
        else
            n_lhs -= length(custom_starts)
        end
        if !(custom_starts[1] isa ComponentArray)
            custom_starts = [ComponentArray(start, axes_θ) for start in custom_starts]
        end
        #Transform starts to logscale
        partial_transform_to_logscale!.(custom_starts, logscale = logscale)
        for i in eachindex(custom_starts)
            #ensure starts satisfy bounds
            starts[row_idx+i-1, :] .= clamp.(custom_starts[i], lb, ub)
        end
        row_idx += length(custom_starts)
    end

    if n_lhs > 0
        rng = isnothing(multistart_seed) ? Random.default_rng() : Random.MersenneTwister(multistart_seed)
        starts[row_idx:end, :] .= latin_hypercube_samples(n_lhs, lower, upper; rng = rng)
        #for generated starts set noise params at indices to passed in function call
        for entry in noise_params
            if !(1 <= entry[1] <= d)
                if !isnothing(lb) && (lb[entry[1]] > entry[2] || ub[entry[1]] < entry[2])
                    error("Passed noise parameter value is outside of bounds")
                else
                    starts[row_idx:end, entry[1]] .= entry[2]
                end
            else
                error("Noise parameter index outside of parameter vector")
            end
        end
    end
    if vectorised
        starts_list = [vec(starts[i, :]) for i in 1:n_starts]
    else
        starts_list = [ComponentArray(vec(starts[i, :]), p.axes_θ) for i in 1:n_starts]
    end
    #If prefiltering sort starts by likelihood value
    if !isnothing(prefilter)
        if n_lhs < n_starts
            starts_keep = starts_list[1:(n_starts - n_lhs)]
            starts_list = starts_list[(n_starts - n_lhs + 1):end]
        else 
            starts_keep = nothing
        end
        if vectorised 
            sort!(starts_list, by=(x -> get_negloglikelihood_vectorised(x, p)))
        else
            sort!(starts_list, by=(x -> get_negloglikelihood(x, p)))
        end
        #ensure keep initial and custom starts if wanted
        if !isnothing(starts_keep)
            starts_list = [starts_keep; starts_list[1:(max(1,multistart)-length(starts_keep))]]
        else
            starts_list = starts_list[1:max(1,multistart)]
        end
    end
    n_starts = max(1,multistart)

    best_raw_any = nothing
    best_start_any = 1
    best_raw_finite = nothing
    best_obj_finite = Inf
    best_start_finite = 1
    best_raw_success_finite = nothing
    best_obj_success_finite = Inf
    best_start_success_finite = 1

    solutions = Array{SciMLBase.AbstractNoTimeSolution}(undef, n_starts)
    #Start threads, divide n_starts onto maximal thread number
    starts_per_thread = Int(ceil(n_starts/thread_num))
    Threads.@threads for j in 1:min(n_starts, thread_num)
        for i in ((j-1)*starts_per_thread+1):min(j*starts_per_thread, n_starts)
            #Create OptimisationProblem with start and bounds (might be nothing)
            problem = OptimizationProblem(objective, starts_list[i], p, lb=lb, ub=ub)
            solutions[i] = solve(problem, solver_optim, maxiters = maxiters, maxtime = maxtime, store_trace = store_trace, solver_options...)
        end
    end
    for i in 1:n_starts
        if isnothing(best_raw_any)
            best_raw_any = solutions[i]
            best_start_any = i
        end
        finite_i = isfinite(solutions[i].objective)
        if finite_i
            obj_i = solutions[i].objective
            if isnothing(best_raw_finite) || obj_i < best_obj_finite
                best_raw_finite = solutions[i]
                best_obj_finite = obj_i
                best_start_finite = i
            end
            if SciMLBase.successful_retcode(solutions[i].retcode) && (isnothing(best_raw_success_finite) || obj_i < best_obj_success_finite)
                best_raw_success_finite = solutions[i]
                best_obj_success_finite = obj_i
                best_start_success_finite = i
            end
        end
    end
    if !isnothing(best_raw_success_finite)
        estimate_raw = best_raw_success_finite
        best_start_idx = best_start_success_finite
    elseif !isnothing(best_raw_finite)
        estimate_raw = best_raw_finite
        best_start_idx = best_start_finite
    else
        estimate_raw = best_raw_any
        best_start_idx = best_start_any
    end
    if vectorised
        estimate_u = ComponentArray(estimate_raw.u, p.axes_θ)
    else 
        estimate_u = estimate_raw.u
    end

    for i in 1:n_starts
        if !(SciMLBase.successful_retcode(solutions[i].retcode)) && (solutions[i].objective < estimate_raw.objective)
            @warn "There is a failed optimisation with a better final objective value."
            println("Note all parameters are given in the transformed space. Original output of solver:")
            println(solutions[i].original)
        end
    end
    #println(starts_list)
    #println(solutions)
    #transform parameters back into non logscale
    partial_transform_to_logscale!(estimate_u, logscale = logscale, detransform = true)
    if run_CI
        CI = inverse_hessian(estimate.u, p, confidence = confidence, logscale = logscale, finite_not_forward = finite_not_forward, sandwich = sandwich)
        estimate = (u = estimate_u, retcode = estimate_raw.retcode, objective = estimate_raw.objective, raw = estimate_raw, CI = CI, multistart_best_start = best_start_idx, multistart_nstarts = n_starts)
    else
        estimate = (u = estimate_u, retcode = estimate_raw.retcode, objective = estimate_raw.objective, raw = estimate_raw, multistart_best_start = best_start_idx, multistart_nstarts = n_starts)
    end
    if printing
        println("Estimate: ", estimate_u)
        println("Retcode: ", estimate.retcode)
        println("Objective: ", estimate.objective)
        println("Multistart best start: ", best_start_idx, " Number of Starts: ", n_starts)
    end
    return estimate
end

#prior that is log of default value 1
function default_log_prior(x)
    return 0
end

#For LogTargetDensity Interface to use in sampled optimiser
struct LogTargetDensity{T<:NamedTuple, T2<:AbstractVector, T3<:Function} 
    p::T #for evaluating negloglikelihood
    lb::T2 #lower bound for checking if point admissable in likelihood
    ub::T2 #upper bound for checking if point admissable in likelihood
    prior::T3 #log prior function
end

LogDensityProblems.logdensity(p::LogTargetDensity, θ) = (all(p.ub .>= θ .>= p.lb)) ? (-get_negloglikelihood_vectorised(θ, p.p) + p.prior(θ)) : -Inf 
LogDensityProblems.dimension(p::LogTargetDensity) = length(p.lb)
LogDensityProblems.capabilities(::LogTargetDensity) = LogDensityProblems.LogDensityOrder{0}()

function optimise_sampled(m::FullModel, data::Tuple; per_chain::Int64 = 10^4, nadapts::Int64 = 0, bound_abs::Union{Nothing, AbstractFloat} = nothing, lower_upper::Union{Nothing, Tuple{ComponentArray, ComponentArray}} = nothing,
    objective_fail_hard::Bool = false, objective_warn::Bool = true, multistart::Int = 1, max_threads::Int = multistart, multistart_seed::Union{Nothing, Int} = nothing, multistart_include_initial::Bool = true, multistart_bounds::Union{Nothing, Tuple{AbstractVector, AbstractVector}, AbstractFloat} = nothing, 
    use_model_bounds::Bool = true, prefilter::Union{Int, Nothing} = nothing, custom_starts::Union{AbstractVector,Nothing} = nothing, noise_params::Tuple{Vararg{Tuple{Int, AbstractFloat}}} = (), printing::Bool = true, run_CI::Bool = false, confidence::AbstractFloat = 0.95,
    sampler = nothing, prior::Union{Distribution, Function, Nothing} = nothing, sampling_options = (), sampling_rng::AbstractRNG = Xoshiro(42), ODE_options = (AutoTsit5(Rosenbrock23()),))
    
    names = get_keys_PK(m.pk_model)
    #create ODE problem for each person in data
    sys = create_ode_system(m.pk_model)
    problems = Tuple(create_problem(m.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time[2])) for person in data)
    #create initial guess
    θ_0 = ComponentArray((PK = m.pk_model.θ, Seizure = m.seizure_model.θ)) 
    #get indices for setting θ
    indices_θ = [ModelingToolkit.parameter_index(sys, x).idx for x in keys(θ_0.PK)]
    #ensure nadapts and per_chain are valid
    per_chain = max(1,per_chain)
    nadapts = max(0, nadapts)
    
    axes_θ = getaxes(θ_0)
    labels_θ = labels(θ_0)
    θ_0_vec = collect(θ_0)
    d = length(θ_0_vec)
    #Set a default sampler for the given dimension
    if isnothing(sampler)
        sampler = RWMH(MvNormal(zeros(d), I))
    end

    #set bounds if required, handle if both individual and absolute bounds
    if !isnothing(lower_upper)
        lb, ub = lower_upper
        if length(ub) != d || length(lb) != d
            error("Upper and lower bounds must match parameter dimension $d")
        end
        if !isnothing(bound_abs)
            ub .=  min.(ub, bound_abs)
            if any(ub .< -bound_abs)
                error("Upper bounds too low to fulfill absolute bounds")
            end
            lb .=  max.(lb, -bound_abs)
            if any(lb .> bound_abs)
                error("Lower bounds too high to fulfill absolute bounds")
            end
        end
        #Check if bounds are valid
        if any(ub .< lb)
            error("Upper bounds strictly smaller than lower ones")
        end
    else
        if !isnothing(bound_abs)
            ub = ComponentArray([bound_abs for i in eachindex(θ_0)], getaxes(θ_0))
            lb = ComponentArray([-bound_abs for i in eachindex(θ_0)], getaxes(θ_0))
        else
            ub = nothing
            lb = nothing
        end
    end
    #Check if want to use model bounds and if they are defined
    if use_model_bounds
        if hasproperty(m.pk_model, :bounds) || hasproperty(m.seizure_model, :bounds)
            if !hasproperty(m.pk_model, :bounds)
                lb_model = ComponentArray(PK = ComponentArray([-Inf for e in m.pk_model.θ], getaxes(m.pk_model.θ)), Seizure = m.seizure_model.bounds.lb)
                ub_model = ComponentArray(PK = ComponentArray([Inf for e in m.pk_model.θ], getaxes(m.pk_model.θ)), Seizure = m.seizure_model.bounds.ub)
            elseif !hasproperty(m.seizure_model, :bounds)
                lb_model = ComponentArray(PK = m.pk_model.bounds.lb, Seizure = ComponentArray([-Inf for e in m.seizure_model.θ], getaxes(m.seizure_model.θ)))
                ub_model = ComponentArray(PK = m.pk_model.bounds.ub, Seizure = ComponentArray([Inf for e in m.seizure_model.θ], getaxes(m.seizure_model.θ)))
            else
                lb_model = ComponentArray(PK = m.pk_model.bounds.lb, Seizure = m.seizure_model.bounds.lb)
                ub_model = ComponentArray(PK = m.pk_model.bounds.ub, Seizure = m.seizure_model.bounds.ub)
            end
            #Check internal consistency and with lb, ub if defined
            if length(ub_model) != d || length(lb_model) != d
                error("Upper and lower model bounds must match parameter dimension $d")
            end
            if any(ub_model .< lb_model)
                error("Upper Model bounds strictly smaller than lower ones")
            end
            if !isnothing(lb)
                ub .=  min.(ub, ub_model)
                lb .=  max.(lb, lb_model)
                if any(lb .> ub)
                    error("Model and manual bounds cannot be satisfied at the same time")
                end
            else
                lb = lb_model
                ub = ub_model
            end
        else
            lb_model = nothing
            ub_model = nothing
        end
    end

    #Ensure initial guess satifies bounds
    if !isnothing(lb)
        θ_0_vec .= clamp.(θ_0_vec, lb, ub)
    end

    #Do everything vectorised, not sure if samplers can handle ComponentArrays
    n_starts = max(multistart, 1)
    thread_num = max(max_threads, 1)
    internal_threads = Int(floor(thread_num/min(n_starts, thread_num)))
    p = (m = m, data = data, logscale = (), options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ, cont_seizure = (m.seizure_model isa SeizureModelContinuous), axes_θ = axes_θ, max_threads = internal_threads, objective_fail_hard = objective_fail_hard, objective_warn = objective_warn, objective_warned_ref = Ref(false))
    
    #Change everything to vectors
    ub = collect(ub)
    lb = collect(lb)
    θ_0 = θ_0_vec

    #Define Model with LogDensityProblems interface
    if isnothing(prior)
        model = LogTargetDensity(p, lb, ub, default_log_prior)
    elseif prior isa Function
        model = LogTargetDensity(p, lb, ub, (log ∘ prior))
    else
        func_prior(x) = logpdf(prior,x)
        model = LogTargetDensity(p, lb, ub, func_prior)
    end
    model = AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(AutoForwardDiff(), model))

    #Assemble multistarts
    lower = zeros(Float64, d)
    upper = zeros(Float64, d)
    if !isnothing(multistart_bounds)
        if typeof(multistart_bounds) <: AbstractFloat
            lower .= Float64(-multistart_bounds)
            upper .= Float64(multistart_bounds)
        else
            lower_raw, upper_raw = multistart_bounds
            if length(lower_raw) != d || length(upper_raw) != d
                error("multistart_bounds must match parameter dimension $d")
            end
            lower .= Float64.(lower_raw)
            upper .= Float64.(upper_raw)
        end
    #check if lb, ub are defined and finite, try those instead
    elseif !isnothing(lb) && all(isfinite.(lb)) && all(isfinite.(ub))
        lower = lb
        upper = ub
    else
        #Fallback finite box around initial point in unconstrained mode.
        lower .= Float64.(θ_0_vec) .- 2.0
        upper .= Float64.(θ_0_vec) .+ 2.0
    end

    #ensure generated starts will satisfy bounds
    if !isnothing(lb)
        lower .= max.(lower, lb)
        upper .= min.(upper, ub)
    end
    if any(.!isfinite.(lower)) || any(.!isfinite.(upper)) || any(upper .<= lower)
        error("Invalid multistart bounds: require finite values and upper > lower component-wise")
    end

    if !isnothing(prefilter)
        if prefilter <= n_starts
            error("Amount of starts for prefiltering must be strictly greater than multistarts")
        else
            n_starts = prefilter
        end
    end
    starts = Matrix{Float64}(undef, n_starts, d)
    row_idx = 1
    if multistart_include_initial
        starts[row_idx, :] .= Float64.(θ_0_vec)
        row_idx += 1
    end
    n_lhs = n_starts - (multistart_include_initial ? 1 : 0)

    #Check if passed starts work
    if !isnothing(custom_starts) && !isempty(custom_starts)
        if any([length(start) != d for start in custom_starts])
            error("Passed custom starts do not match parameter dimension")
        end
        if length(custom_starts) > max(multistart,1) - (multistart_include_initial ? 1 : 0)
            error("Too many starts passed for multistart")
        else
            n_lhs -= length(custom_starts)
        end
        if !(custom_starts[1] isa ComponentArray)
            custom_starts = [ComponentArray(start, axes_θ) for start in custom_starts]
        end
        for i in eachindex(custom_starts)
            #ensure starts satisfy bounds
            starts[row_idx+i-1, :] .= clamp.(custom_starts[i], lb, ub)
        end
        row_idx += length(custom_starts)
    end

    if n_lhs > 0
        rng = isnothing(multistart_seed) ? Random.default_rng() : Random.MersenneTwister(multistart_seed)
        starts[row_idx:end, :] .= latin_hypercube_samples(n_lhs, lower, upper; rng = rng)
        #for generated starts set noise params at indices to passed in function call
        for entry in noise_params
            if !(1 <= entry[1] <= d)
                if !isnothing(lb) && (lb[entry[1]] > entry[2] || ub[entry[1]] < entry[2])
                    error("Passed noise parameter value is outside of bounds")
                else
                    starts[row_idx:end, entry[1]] .= entry[2]
                end
            else
                error("Noise parameter index outside of parameter vector")
            end
        end
    end

    starts_list = [vec(starts[i, :]) for i in 1:n_starts]

    #If prefiltering sort starts by likelihood value
    if !isnothing(prefilter)
        if n_lhs < n_starts
            starts_keep = starts_list[1:(n_starts - n_lhs)]
            starts_list = starts_list[(n_starts - n_lhs + 1):end]
        else 
            starts_keep = nothing
        end
        sort!(starts_list, by=(x -> get_negloglikelihood_vectorised(x, p)))
        #ensure keep initial and custom starts if wanted
        if !isnothing(starts_keep)
            starts_list = [starts_keep; starts_list[1:(max(1,multistart)-length(starts_keep))]]
        else
            starts_list = starts_list[1:max(1,multistart)]
        end
    end
    n_starts = max(1,multistart)

    #Start threads, divide n_starts onto maximal thread number
    not_parallel = Int(ceil(n_starts/thread_num))
    chains = Array{Chains}(undef, not_parallel)
    #For recording sampling times rather naively, for this interface cant get it to log them
    times = Array{Real}(undef, not_parallel)
    for j in 1:not_parallel
        starts_thread = [starts_list[i] for i in ((j-1)*thread_num+1):min(j*thread_num, n_starts)]
        println("Starts: ", starts_thread, length(starts_thread))
        println
        start = time()
        chains[j] = sample(sampling_rng, model, sampler, MCMCThreads(), (per_chain+nadapts), length(starts_thread); n_adapts = nadapts, param_names=labels_θ, initial_params = starts_thread, chain_type=Chains, sampling_options...)
        ending = time()
        times[j] = ending - start
    end
    #Collect all chains into one object
    Chain = cat(chains...; dims = 3)

    means = ComponentArray([mean(Chain, label) for label in labels_θ], axes_θ)
    eval = get_negloglikelihood(means, p)

    if run_CI
        if !(0 ≤ confidence ≤ 1)
            @warn "Invalid confidence for CIs, reset to default 0.95"
            confidence = 0.95
        end
        confidence = 1-confidence
        CI = ComponentArray([quantile(vec(Chain[label]), [confidence/2, 1-confidence/2]) for label in labels_θ], axes_θ)
        estimate = (u = means, objective = eval, raw = Chain, times = times, CI = CI, multistart_nstarts = n_starts)
    else
        estimate = (u = means, objective = eval, raw = Chain, times = times, multistart_nstarts = n_starts)
    end
    if printing
        println("Estimate: ", estimate.u)
        println("Objective: ", estimate.objective)
    end

    return estimate
end

#finite_not_forward allows to switch to finite_diff hessian instead of ForwardDiff, often faster but less accurate
function inverse_hessian(θ::ComponentArray, m::FullModel, data::Tuple; confidence::AbstractFloat = 0.95, logscale::Tuple{Vararg{String}} = (), finite_not_forward::Bool = false, sandwich::Bool = true, ODE_options = (AutoTsit5(Rosenbrock23()),))
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    problems = Tuple(create_problem(m.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time[2])) for person in data)
    indices_θ = [ModelingToolkit.parameter_index(sys, x).idx for x in keys(θ.PK)]
    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ, cont_seizure = (m.seizure_model isa SeizureModelContinuous))
    return inverse_hessian(θ, p, confidence = confidence, logscale = logscale, finite_not_forward=finite_not_forward, sandwich = sandwich)
end

function inverse_hessian(θ::ComponentArray, p::NamedTuple; confidence::AbstractFloat = 0.95, logscale::Tuple{Vararg{String}} = (), finite_not_forward::Bool = false, sandwich::Bool = true)
    #check whether accidentally entered percentage instead of confidence in (0,1)
    if confidence>1
        confidence = confidence/100
        @warn "Presumably you meant a percentage. Your input confidence has been divided by 100"
    end
    if !(0 ≤ confidence ≤ 1)
        @warn "Invalid confidence for CIs, reset to default 0.95"
        confidence = 0.95
    end
    f(x) = get_negloglikelihood(x, p)
    θ_use = deepcopy(θ)
    #transform as specified
    partial_transform_to_logscale!(θ_use, logscale = logscale)
    #println("Gradient: ", ForwardDiff.gradient(f,θ_use))
    if !(finite_not_forward)
        #This takes very long
        H = ForwardDiff.hessian(f,θ_use)
    else
        H = FiniteDiff.finite_difference_hessian(f, θ_use)
    end
    #Fallback definition of bounds if sth doesn't work
    bounds = [(-Inf, Inf) for i in eachindex(θ_use)]
    bounds_sandwich = [(-Inf, Inf) for i in eachindex(θ_use)]
    try
        #Set bounds for simple inverse hessian CI
        H_inv = inv(H)
        #println(H_inv)
        positive_diagonal = true
        for i in eachindex(θ_use)
            positive_diagonal = (0 ≤ H_inv[i,i]) && positive_diagonal
        end
        if positive_diagonal
            q = quantile(Normal(), (1-(1-confidence)/2))
            #By symmetry other one is just the negative
            #Note q>= 0 since quantile of standard normal positive for >=0.5, ensured for confidence<=1
            bounds = [(θ_use[i] - sqrt(H_inv[i,i])*q, θ_use[i] + sqrt(H_inv[i,i])*q) for i in eachindex(θ_use)]
        else
            @warn "Negative diagonal entry in inverse hessian"
        end
        #Set bounds for sandwich CI
        if sandwich
            Bread = H_inv
            p_indices_not_data = Tuple(setdiff(keys(p), (:data,)))
            p_per_person = Tuple(merge(NamedTuple{p_indices_not_data}(p), (data = (person,),)) for person in p.data)
            if !(finite_not_forward)
                grads = Tuple(ForwardDiff.gradient(x -> get_negloglikelihood(x, p_person),θ_use) for p_person in p_per_person)
            else
                grads = Tuple(ForwardDiff.finite_difference_gradient(x -> get_negloglikelihood(x, p_person),θ_use) for p_person in p_per_person)
            end 
            Meat = sum(Tuple(grad * (grad') for grad in grads))
            #println("Bread= ", Bread)
            #println("Meat= ", Meat)
            Sandwich = Bread*Meat*Bread
            positive_diagonal = true
            for i in eachindex(θ_use)
                positive_diagonal = (0 ≤ Sandwich[i,i]) && positive_diagonal
            end
            if positive_diagonal
                q = quantile(Normal(), (1-(1-confidence)/2))
                #By symmetry other one is just the negative
                #Note q>= 0 since quantile of standard normal positive for >=0.5, ensured for confidence<=1
                bounds_sandwich = [(θ_use[i] - sqrt(Sandwich[i,i])*q, θ_use[i] + sqrt(Sandwich[i,i])*q) for i in eachindex(θ_use)]
            else
                @warn "Negative diagonal entry in sandwich estimator"
            end
        end
    catch e
        if e isa LinearAlgebra.SingularException
            @warn "Calculated Hessian is singular"
        elseif e isa ArgumentError
            @warn "Issue with calculated hessian: $(e)"
        else
            rethrow(e)
        end
    end
    #now assign intervals to correct keys
    if sandwich
        CIs = (InverseHessian = ComponentArray(bounds, getaxes(θ_use)), Sandwich = ComponentArray(bounds_sandwich, getaxes(θ_use)))
    else 
        CIs = (InverseHessian = ComponentArray(bounds, getaxes(θ_use)),)
    end
    #println("CI untransformed: ", CI)
    #transform logscale ones, can't use partial transform since entries are now tuples, have to broadcast
    for CI in CIs    
        for label in logscale
            if label in labels(CI.PK) || Symbol(label) in keys(CI.PK)
                indices = label2index(CI.PK,label)
                for index in indices
                    @inbounds CI.PK[index] = exp.(CI.PK[index])
                end
            end

            if label in labels(CI.Seizure) || Symbol(label) in keys(CI.Seizure)
                indices = label2index(CI.Seizure,label)
                for index in indices
                    @inbounds CI.Seizure[index] = exp.(CI.Seizure[index])
                end
            end
        end
    end
    return CIs
end

#m determines model parts, n determines number of people, timepoints for measurements
function generate_data(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; timepoints_PK::AbstractVector = 0:14.0:time, timepoints_seizure::AbstractVector = 0:m.seizure_model.timeframe.inherent_timeframe:time, max_threads::Int = 1, max_events::Union{Int, Nothing} = nothing, just_Bool::Bool = false, generate_in_lumps::Bool = true, wo_treatment::AbstractFloat = 3.0, ODE_options = (AutoTsit5(Rosenbrock23()),))
    if max(timepoints_PK..., timepoints_seizure...)>time
        error("Timepoints for measurements occuring after assigned timeframe")
    end
    population = generate_population(m.population_gen, n)
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    people_per_thread = Int(ceil(n/max_threads))
    seeds = rand(Int64,n)
    Threads.@threads for j in 1:min(n, max_threads)
        for i in ((j-1)*people_per_thread+1):min(j*people_per_thread, n)
            #For basic calculations guarantees reproducibility when run in same part of seeded program
            Random.seed!(seeds[i])
            assign_dose!(m.dose_gen, population[i], names = names, timeframe = time, wo_treatment = wo_treatment)
            sol = generate_measurements!(m.pk_model, sys, population[i], timepoints = timepoints_PK, endpoint = time, options = ODE_options)
            if m.seizure_model isa SeizureModelDiscrete
                generate_seizures!(m.seizure_model, sol, population[i], timepoints = timepoints_seizure, just_Bool = just_Bool, generate_in_lumps = generate_in_lumps, names=names)
                summarise_seizures!(m.seizure_model, population[i], timepoints = timepoints_seizure, just_Bool= just_Bool)
            else
                generate_seizures!(m.seizure_model, sol, population[i], endpoint=time, max_events=max_events, names=names)
            end
        end
    end
    return population
end

#for later when want to update doses etc regularly
function generate_data_updating(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; update_reg::AbstractFloat = time, timepoints_PK::AbstractVector = 0:14.0:time, timepoints_seizure::AbstractVector = 0:m.seizure_model.timeframe.inherent_timeframe:time, max_threads::Int = 1, max_events::Union{Int, Nothing} = nothing, just_Bool::Bool = false, generate_in_lumps::Bool = true, wo_treatment::AbstractFloat = 3.0, ODE_options = (AutoTsit5(Rosenbrock23()),))
    if max(timepoints_PK..., timepoints_seizure...)>time
        error("Timepoints for measurements occuring after assigned timeframe")
    end
    data = generate_population(m.population_gen, n)
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    people_per_thread = Int(ceil(n/max_threads))
    seeds = rand(Int64,n)
    Threads.@threads for j in 1:min(n, max_threads)
        for i in ((j-1)*people_per_thread+1):min(j*people_per_thread, n)
            #For basic calculations guarantees reproducibility when run in same part of seeded program
            Random.seed!(seeds[i])
            if wo_treatment > 0
                passed_time = min(wo_treatment, time)
            else
                passed_time = min(update_reg, time)
            end
            #here generate for min(wo_treatment,time)
            assign_dose!(m.dose_gen, data[i], names=names, timeframe = passed_time, wo_treatment = wo_treatment)
            current_timepoints_PK = [t for t in timepoints_PK if 0.0 <= t < passed_time] #filter timepoints in this interval
            sol = generate_measurements!(m.pk_model, sys, data[i], timepoints = current_timepoints_PK, endpoint = passed_time, options = ODE_options)
            if m.seizure_model isa SeizureModelDiscrete
                current_timepoints_seizure = [t for t in timepoints_seizure if 0.0 <= t < passed_time]
                generate_seizures!(m.seizure_model, sol, data[i], timepoints=current_timepoints_seizure, just_Bool = just_Bool, generate_in_lumps = generate_in_lumps, names=names)
            else
                generate_seizures!(m.seizure_model, sol, data[i], endpoint=passed_time, max_events=max_events, names=names)
            end
            while passed_time < time
                sol_prev = sol
                increment = min(time, passed_time + update_reg) - passed_time
                passed_time += increment
                current_timepoints_PK = [t for t in timepoints_PK if (passed_time-increment)<= t < passed_time] #filter timepoints in this interval
                current_timepoints_seizure = [timepoints_seizure[i] for i in eachindex(timepoints_seizure) 
                                                    if ((passed_time-increment)<= timepoints_seizure[i] < passed_time || (passed_time-increment)<= timepoints_seizure[min(i+1, length(timepoints_seizure))] <= passed_time)] #capture interval overlap
                start_solution = min(passed_time-increment, current_timepoints_seizure[1])
                assign_dose!(m.dose_gen, data[i], names=names, timeframe = increment)
                sol = generate_measurements!(m.pk_model, sys, data[i], timepoints = current_timepoints_PK, endpoint = passed_time, start = (start_solution, sol_prev(start_solution)), options = ODE_options)
                if m.seizure_model isa SeizureModelDiscrete
                    generate_seizures!(m.seizure_model, sol, data[i], timepoints = current_timepoints_seizure, just_Bool = just_Bool, generate_in_lumps = generate_in_lumps, names=names)
                else
                    #Ensure dont have more than max events
                    if isnothing(max_events)
                        events_left = nothing
                    else
                        events_left = max_events - length(data[i].seizure_counts)
                    end
                    generate_seizures!(m.seizure_model, sol, data[i], endpoint=passed_time, start = (passed_time - increment), max_events=events_left, names=names)
                end
            end
            if m.seizure_model isa SeizureModelDiscrete
                summarise_seizures!(m.seizure_model, data[i], timepoints = timepoints_seizure, just_Bool= just_Bool)
            end
        end
    end
    return data
end

#generate with a parameter modified with randomly drawn addition for specified indices and distributions, effects constant per person
#is automatically updating, if no update_reg is passed then no updates
#expects modifications as tuple of 2-tuples (index, distribution to add)
function generate_data_modified(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; modifications::Tuple{Vararg{Tuple{Int, Union{Distribution, Number, Function}}}}, update_reg::AbstractFloat = time, timepoints_PK::AbstractVector = 0:14.0:time, timepoints_seizure::AbstractVector = 0:m.seizure_model.timeframe.inherent_timeframe:time, max_events::Union{Int, Nothing} = nothing, max_threads::Int = 1, just_Bool::Bool = false, generate_in_lumps::Bool = true, wo_treatment::AbstractFloat = 0.0, ODE_options = (AutoTsit5(Rosenbrock23()),))
    #Check modification if functions have correct return type
    if any([(mod[2] isa Function && !(mod[2](1.0) isa Union{Number, Distribution})) for mod in modifications])
        error("Modification function has unsupported return type")
    end
    if !allunique(Tuple(mod[1] for mod in modifications)) 
        error("Two modifications passed for same parameter index")
    end 
    current_θ = ComponentArray(PK = m.pk_model.θ, Seizure = m.seizure_model.θ)
    modification_unfunctioned = (mod[2] isa Function ? (mod[1], mod[2](current_θ[mod[1]])) : mod for mod in modifications)
    modifiers = Tuple(mod[2] isa Number ? (mod[1], Dirac(mod[2])) : mod for mod in modification_unfunctioned)
    data = Vector{Person}(undef, n)
    people_per_thread = Int(ceil(n/max_threads))
    seeds = rand(Int64, n)
    Threads.@threads for j in 1:min(n, max_threads)
        for i in ((j-1)*people_per_thread+1):min(j*people_per_thread, n)
            #For basic calculations guarantees reproducibility when run in same part of seeded program
            Random.seed!(seeds[i])
            modifiers_person = [(mod[1],rand(mod[2])) for mod in modifiers]
            new_θ = ComponentArray(PK = m.pk_model.θ, Seizure = m.seizure_model.θ)
            for mod in modifiers_person
                new_θ[mod[1]] = mod[2]
            end
            new_model = deepcopy(m) #create model with modified θ here
            new_model.pk_model.θ .= new_θ.PK
            new_model.seizure_model.θ .= new_θ.Seizure
            #generate 1 person population with these parameters
            person = generate_data_updating(new_model, 1, time, update_reg = update_reg, timepoints_PK = timepoints_PK, timepoints_seizure = timepoints_seizure, max_events = max_events, just_Bool = just_Bool, generate_in_lumps = generate_in_lumps, wo_treatment = wo_treatment, ODE_options = ODE_options)
            data[i] = person[1]
            #write modifiers to person
            append!(person[1].random_effects, modifiers_person)
        end
    end
    return Tuple(data)
end

function plot_fit(mod::FullModel, data::Tuple; true_param::Union{ComponentArray, Nothing} = ComponentArray(PK = mod.pk_model.θ, Seizure = mod.seizure_model.θ), estimate_param::Union{ComponentArray, Nothing} = nothing,
    individuals::AbstractVector = [1], endpoint::Union{AbstractFloat, Nothing} = nothing, time_pk::Union{Tuple{Union{Int, AbstractFloat}, Union{Int, AbstractFloat}}, AbstractFloat, Int, Nothing} = nothing, 
    time_seizures::Union{Tuple{Union{Int, AbstractFloat}, Union{Int, AbstractFloat}}, AbstractFloat, Int} = 10, samples_seizures::Int = 1000, display_plot::Bool = true, options = (AutoTsit5(Rosenbrock23()),))

    output = Plots.Plot[]
    if isnothing(endpoint)
        measurements_ends = Tuple(person.measurements[end].timepoint for person in data)
        seizure_ends = Tuple(person.seizure_counts[end].time[2] for person in data)
        endpoint = max(measurements_ends...,seizure_ends...)
    end
    if isnothing(time_pk)
        time_pk = (0.0, endpoint)
    end
    if !isnothing(true_param)
        sols = [solve_PK(mod.pk_model, true_param.PK, data[i], endpoint = endpoint, options = options) for i in eachindex(data)]
        if length(data)>0 && !isempty(data[1].random_effects)
            person_param = [deepcopy(true_param) for person in data]
            for i in eachindex(data)
                for mod in data[i].random_effects
                    person_param[i][mod[1]] += mod[2]
                end
            end
            sols_mod = [solve_PK(mod.pk_model, person_param[i].PK, data[i], endpoint = endpoint, options = options) for i in eachindex(data)]
        else
            sols_mod = nothing
        end
    else
        sols = nothing
        sols_mod = nothing
    end
    if !isnothing(estimate_param)
        sols2 = [solve_PK(mod.pk_model, estimate_param.PK, data[i], endpoint = endpoint, options = options) for i in eachindex(data)]
    else 
        sols2 = nothing
    end

    pk_output = plot_fit(mod.pk_model, data, sols_true = sols, sols_estimated = sols2, sols_modified = sols_mod, individuals = individuals, time = time_pk, display_plot = display_plot)
    append!(output, pk_output)
    if isnothing(estimate_param)
        estimate_seizure = nothing
    else 
        estimate_seizure = estimate_param.Seizure
    end
    seizure_output = plot_fit(mod.seizure_model, data, estimate_param = estimate_seizure, sols_true = sols, sols_estimated = sols2, sols_modified = sols_mod, length_PK = length(true_param.PK), names = mod.pk_model.keys, individuals = individuals, time = time_seizures, sample_nr = samples_seizures, display_plot = display_plot)
    append!(output, seizure_output)
    return output
end

function multi_data_run(mod::FullModel, data::Function, estimate::Function, eval::Function; run_count::Int=50, max_threads_runs::Int=1)
    #sample seeds of specified number
    seeds = rand(Int64, run_count)
    #create object to store results
    datas = Vector{Tuple{Vararg{Person}}}(undef, run_count)
    estimates = Vector{NamedTuple}(undef, run_count)
    abs_errors = Vector{ComponentArray}(undef, run_count)
    rel_errors = Vector{ComponentArray}(undef, run_count)
    mean_squared_errors = Vector{Real}(undef, run_count)
    rel_squared_errors =  Vector{Real}(undef, run_count)
    times = Vector{Real}(undef, run_count)
    obj_diffs = Vector{Real}(undef, run_count)
    Input_θ = ComponentArray(PK = mod.pk_model.θ, Seizure = mod.seizure_model.θ)
    runs_per_thread = Int(ceil(run_count/max_threads_runs))
    Threads.@threads for j in 1:min(run_count, max_threads_runs)
        for i in ((j-1)*runs_per_thread+1):min(j*runs_per_thread, run_count)
            #For basic calculations guarantees reproducibility when run in same part of seeded program
            Random.seed!(seeds[i])
            #create data
            data = data()
            datas[i] = data
            #run optimisation
            test_mod = FullModel(typeof(mod.pk_model).name.wrapper(), typeof(mod.seizure_model).name.wrapper(mod.pk_model), deepcopy(mod.population_gen), deepcopy(mod.dose_gen))
            estimate = estimate(test_mod, data)
            #Start setting our objects
            estimates[i] = estimate
            #Calculate errors
            errors_rel = deepcopy(Input_θ)
            errors_abs = deepcopy(Input_θ)
            for i in eachindex(errors_rel)
                errors_rel[i] = abs(estimate.u[i] - Input_θ[i])/abs(Input_θ[i])
            end
            for i in eachindex(errors_abs)
                errors_abs[i] = abs(estimate.u[i] - Input_θ[i])
            end
            abs_errors[i] = errors_abs
            rel_errors[i] = errors_rel
            #For objective check if are estimating Hierarchical
            if hasproperty(estimate, :estimate_PK)
                #handle Hierarchical
                obj_diffs[i] = (PK = (estimate.objective.PK - eval(data).PK), Seizure = estimate.objective.Seizure - eval(data).Seizure)
            else
                obj_diffs[i] = estimate.objective - eval(data)
            end
            mean_squared_errors[i] = sum(errors_abs.^2)/length(Input_θ)
            rel_squared_errors[i] = sum(errors_rel.^2)/length(Input_θ)
            if hasproperty(estimate, :estimate_PK)
                #handle time for hierarchical
                times[i] = Optim.time_run(estimate.estimate_PK.original) + Optim.time_run(estimate.estimate_Seizure.original)
            elseif hasproperty(estimate.raw, :original)
                times[i] = Optim.time_run(estimate.raw.original)
            else
                #handle sampled time
                chain_time = MCMCChains.compute_duration(estimate.raw)
                if chain_time isa Real
                    times[i] = chain_time
                elseif hasproperty(estimate, :times)
                    times[i] = max(estimate.times...)
                else
                    times[i] = NaN
                end
            end
        end
    end
    if !isempty(estimates) && hasproperty(estimates[1], :CI)
        CIs = [estimate.CI for estimate in estimates]
        result = (datas = datas, estimates = estimates, abs_errors = abs_errors, rel_errors = rel_errors, mean_squared_errors = mean_squared_errors, rel_squared_errors = rel_squared_errors, times = times, obj_diffs = obj_diffs, CIs = CIs)
    else
        result = (datas = datas, estimates = estimates, abs_errors = abs_errors, rel_errors = rel_errors, mean_squared_errors = mean_squared_errors, rel_squared_errors = rel_squared_errors, times = times, obj_diffs = obj_diffs)
    end
    return result
    #plot (relative) mean squared error, plot times perhaps -> maybe plotting doesnt make sense here yet, need longitudinal data points
end

end # module EpilepsyModels