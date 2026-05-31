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

export optimise, optimise_hierarchical, generate_data, generate_data_updating, get_negloglikelihood_evaluated, get_negloglikelihood_evaluated_hierarchical, plot_fit,
BasicDoses, PolyDosesRandom, PolyDoses, PKBasic, PKLEV, PKLEVNoAbsorption, PKCBZ, PKVPA, PKLTG, PKBigFour,
BasicPersonGenerator, PersonGeneratorLEV, BigFourPersonGenerator, SeizureBasic, SeizureNegativeBinomial, FullModel

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
                @inbounds loglikelihoods[i] += get_seizure_loglikelihood(θ_use.Seizure, m.seizure_model, sol, data[i], names=names)
            end
        end
        if !keep_going[]
            return Inf
        end
        return -sum(loglikelihoods)
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
    for i in eachindex(data)
        sol = solutions[i]
        @inbounds loglikeli += get_seizure_loglikelihood(θ_use, m.seizure_model, sol, data[i], names=names)
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
    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ)
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

function optimise_hierarchical(m::FullModel, data::Tuple; maxiters::Int64 = 10^4, logscale::Tuple{Vararg{String}} = (), inv_hess_CI::Bool = false, bound_abs::Union{Nothing, AbstractFloat} = nothing, lower_upper::Union{Nothing, Tuple{ComponentArray, ComponentArray}} = nothing, 
                objective_fail_hard::Bool = false, objective_warn::Bool = true, store_trace::Bool = false, solver_optim = LBFGS(linesearch = LineSearches.BackTracking()), ODE_options = (AutoTsit5(Rosenbrock23()),))
    
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
    estimate_PK = solve(problem_PK, solver_optim, maxiters = maxiters, store_trace = store_trace) 
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
    estimate_Seizure = solve(problem_Seizure, solver_optim, maxiters = maxiters, store_trace = store_trace) 
    #transform parameters back into non logscale
    partial_transform_to_logscale_partwise!(estimate_Seizure.u, logscale = logscale, detransform = true)

    estimate = (u = ComponentVector(PK = estimate_PK.u, Seizure = estimate_Seizure.u), retcode = estimate_Seizure.retcode, objective = (PK = estimate_PK.objective, Seizure = estimate_Seizure.objective), estimate_PK = estimate_PK, estimate_Seizure = estimate_Seizure)

    println("Estimate: ", estimate.u)
    println(estimate.retcode)
    println(estimate.objective)
    if inv_hess_CI
        CI = inverse_hessian(estimate.u, p, logscale = logscale)
        return estimate, CI
    else
        return estimate
    end
end

function optimise(m::FullModel, data::Tuple; maxiters::Int64 = 10^4, maxtime::AbstractFloat = Inf, logscale::Tuple{Vararg{String}} = (), inv_hess_CI::Bool = false, bound_abs::Union{Nothing, AbstractFloat} = nothing, lower_upper::Union{Nothing, Tuple{ComponentArray, ComponentArray}} = nothing,
    objective_fail_hard::Bool = false, objective_warn::Bool = true, store_trace::Bool = false, multistart::Int = 1, max_threads::Int = multistart, multistart_seed::Union{Nothing, Int} = nothing, multistart_include_initial::Bool = true, multistart_bounds::Union{Nothing, Tuple{AbstractVector, AbstractVector}, AbstractFloat} = nothing, 
    solver_optim = LBFGS(linesearch = LineSearches.BackTracking()), ODE_options = (AutoTsit5(Rosenbrock23()),))
    
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
    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ, axes_θ = axes_θ, max_threads = internal_threads, objective_fail_hard = objective_fail_hard, objective_warn = objective_warn, objective_warned_ref = Ref(false))
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
    elseif !isnothing(bound_abs)
        #don't use lb and ub here because those will often be infinite in some entries
        lower .= -Float64(bound_abs)
        upper .= Float64(bound_abs)
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

    starts = Matrix{Float64}(undef, n_starts, d)
    row_idx = 1
    if multistart_include_initial
        starts[row_idx, :] .= Float64.(θ_0_vec)
        row_idx += 1
    end
    n_lhs = n_starts - (multistart_include_initial ? 1 : 0)
    if n_lhs > 0
        rng = isnothing(multistart_seed) ? Random.default_rng() : Random.MersenneTwister(multistart_seed)
        starts[row_idx:end, :] .= latin_hypercube_samples(n_lhs, lower, upper; rng = rng)
    end
    if vectorised
        starts_list = [vec(starts[i, :]) for i in 1:n_starts]
    else
        starts_list = [ComponentArray(vec(starts[i, :]), p.axes_θ) for i in 1:n_starts]
    end

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
            solutions[i] = solve(problem, solver_optim, maxiters = maxiters, maxtime = maxtime, store_trace = store_trace)
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
    estimate = (u = estimate_u, retcode = estimate_raw.retcode, objective = estimate_raw.objective, raw = estimate_raw, multistart_best_start = best_start_idx, multistart_nstarts = n_starts)
    println("Estimate: ", estimate_u)
    println("Retcode: ", estimate.retcode)
    println("Objective: ", estimate.objective)
    println("Multistart best start: ", best_start_idx, " Number of Starts: ", n_starts)
    if inv_hess_CI
        CI = inverse_hessian(estimate.u, p, logscale = logscale)
        return estimate, CI
    else
        return estimate
    end
end

#finite_not_forward allows to switch to finite_diff hessian instead of ForwardDiff, often faster but less accurate
function inverse_hessian(θ::ComponentArray, m::FullModel, data::Tuple; confidence::AbstractFloat = 0.95, logscale::Tuple{Vararg{String}} = (), finite_not_forward::Bool = false, sandwich::Bool = true, ODE_options = (AutoTsit5(Rosenbrock23()),))
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    problems = Tuple(create_problem(m.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time[2])) for person in data)
    indices_θ = [ModelingToolkit.parameter_index(sys, x).idx for x in keys(θ.PK)]
    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ)
    return inverse_hessian(θ, p, confidence = confidence, logscale = logscale, finite_not_forward=finite_not_forward, sandwich = sandwich)
end

function inverse_hessian(θ::ComponentArray, p::NamedTuple; confidence::AbstractFloat = 0.95, logscale::Tuple{Vararg{String}} = (), finite_not_forward::Bool = false, sandwich::Bool = true)
    #check whether accidentally entered percentage instead of confidence in (0,1)
    if confidence>1
        confidence = confidence/100
        @warn "Presumably you meant a percentage. Your input confidence has been divided by 100"
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
function generate_data(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; timepoints_PK::AbstractVector = 0:14.0:time, timepoints_seizure::AbstractVector = 0:m.timeframe.inherent_timeframe:time, just_Bool::Bool = false, generate_in_lumps::Bool = true, wo_treatment::AbstractFloat = 3.0, ODE_options = (AutoTsit5(Rosenbrock23()),))
    if max(timepoints_PK..., timepoints_seizure...)>time
        error("Timepoints for measurements occuring after assigned timeframe")
    end
    population = generate_population(m.population_gen, n)
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    for person in population
        assign_dose!(m.dose_gen, person, names = names, timeframe = time, wo_treatment = wo_treatment)
        sol = generate_measurements!(m.pk_model, sys, person, timepoints = timepoints_PK, endpoint = time, options = ODE_options)
        generate_seizures!(m.seizure_model, sol, person, timepoints = timepoints_seizure, just_Bool = just_Bool, generate_in_lumps = generate_in_lumps, names=names)
        summarise_seizures!(m.seizure_model, person, timepoints = timepoints_seizure, just_Bool= just_Bool)
    end
    return population
end

#for later when want to update doses etc regularly
function generate_data_updating(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; update_reg::AbstractFloat = time, timepoints_PK::AbstractVector = 0:14.0:time, timepoints_seizure::AbstractVector = 0:1.0:time, just_Bool::Bool = false, generate_in_lumps::Bool = true, wo_treatment::AbstractFloat = 3.0, ODE_options = (AutoTsit5(Rosenbrock23()),))
    if max(timepoints_PK..., timepoints_seizure...)>time
        error("Timepoints for measurements occuring after assigned timeframe")
    end
    population = generate_population(m.population_gen, n)
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    for person in population
        passed_time = min(wo_treatment, time)
        #here generate for min(wo_treatment,time)
        assign_dose!(m.dose_gen, person, names=names, timeframe = passed_time, wo_treatment = wo_treatment)
        current_timepoints_PK = [t for t in timepoints_PK if 0.0 <= t < passed_time] #filter timepoints in this interval
        sol = generate_measurements!(m.pk_model, sys, person, timepoints = current_timepoints_PK, endpoint = passed_time, options = ODE_options)
        current_timepoints_seizure = [t for t in timepoints_seizure if 0.0 <= t < passed_time]
        generate_seizures!(m.seizure_model, sol, person, timepoints=current_timepoints_seizure, just_Bool = just_Bool, generate_in_lumps = generate_in_lumps, names=names)
        while passed_time < time
            sol_prev = sol
            increment = min(time, passed_time + update_reg) - passed_time
            passed_time += increment
            current_timepoints_PK = [t for t in timepoints_PK if (passed_time-increment)<= t < passed_time] #filter timepoints in this interval
            current_timepoints_seizure = [timepoints_seizure[i] for i in eachindex(timepoints_seizure) 
                                                if ((passed_time-increment)<= timepoints_seizure[i] < passed_time || (passed_time-increment)<= timepoints_seizure[min(i+1, length(timepoints_seizure))] <= passed_time)] #capture interval overlap
            start_solution = min(passed_time-increment, current_timepoints_seizure[1])
            assign_dose!(m.dose_gen, person, names=names, timeframe = increment)
            sol = generate_measurements!(m.pk_model, sys, person, timepoints = current_timepoints_PK, endpoint = passed_time, start = (start_solution, sol_prev(start_solution)), options = ODE_options)
            generate_seizures!(m.seizure_model, sol, person, timepoints = current_timepoints_seizure, just_Bool = just_Bool, generate_in_lumps = generate_in_lumps, names=names)
        end
        summarise_seizures!(m.seizure_model, person, timepoints = timepoints_seizure, just_Bool= just_Bool)
    end
    return population
end

function plot_fit(mod::FullModel, data::Tuple; true_param::Union{ComponentArray, Nothing} = ComponentArray(PK = mod.pk_model.θ, Seizure = mod.seizure_model.θ), estimate_param::Union{ComponentArray, Nothing} = nothing,
    individuals::AbstractVector = [1], endpoint::Union{AbstractFloat, Nothing} = nothing, time_pk::Union{Tuple{Union{Int, AbstractFloat}, Union{Int, AbstractFloat}}, AbstractFloat, Int, Nothing} = nothing, 
    time_seizures::Union{Tuple{Union{Int, AbstractFloat}, Union{Int, AbstractFloat}}, AbstractFloat, Int} = 10, display_plot::Bool = true, options = (AutoTsit5(Rosenbrock23()),))

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
    else
        sols = nothing
    end
    if !isnothing(estimate_param)
        sols2 = [solve_PK(mod.pk_model, estimate_param.PK, data[i], endpoint = endpoint, options = options) for i in eachindex(data)]
    else 
        sols2 = nothing
    end

    pk_output = plot_fit(mod.pk_model, data, sols_true = sols, sols_estimated = sols2, individuals = individuals, time = time_pk, display_plot = display_plot)
    append!(output, pk_output)
    if isnothing(estimate_param)
        estimate_seizure = nothing
    else 
        estimate_seizure = estimate_param.Seizure
    end
    seizure_output = plot_fit(mod.seizure_model, data, estimate_param = estimate_seizure, sols_true = sols, sols_estimated = sols2, names = mod.pk_model.keys, individuals = individuals, time = time_seizures, display_plot = display_plot)
    append!(output, seizure_output)
    return output
end

end # module EpilepsyModels
