module EpilepsyModels

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Optimization
using ForwardDiff
using ComponentArrays
using FiniteDiff
using Parameters
using LinearAlgebra

export optimise, optimise_multistart, generate_data, generate_data_updating, get_negloglikelihood_evaluated, BasicDoses, PolyDosesRandom, PolyDoses, PKBasic, PKLEV, 
PKCBZ, PKVPA, BasicPersonGenerator, PersonGeneratorLEV, BigFourPersonGenerator, SeizureBasic, FullModel

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
    θ_use = copy(θ)
    #for keys in logscale take exponential in θ
    partial_transform_to_logscale!(θ_use, logscale = logscale, detransform = true)
    loglikeli = zero(eltype(θ_use))
    try
        for i in eachindex(data)
            @inbounds sol = solve_PK(problems[i], system, θ_use.PK, indices_θ = indices, options = options)
            if !(SciMLBase.successful_retcode(sol))
                return Inf
            end
            @inbounds loglikeli += get_PK_loglikelihood(θ_use.PK, data[i], sol=sol)
            @inbounds loglikeli += get_seizure_loglikelihood(θ_use.Seizure, m.seizure_model, sol, data[i], names=names)
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

#some optimisers cannot work with componentarrays directly
function get_negloglikelihood_vectorised(θ::AbstractVector, p::NamedTuple)
    θ_struct = ComponentArray(copy(θ), p.axes_θ)
    return get_negloglikelihood(θ_struct, p)
end

function get_negloglikelihood_evaluated(θ::ComponentArray, m::FullModel, data::Tuple; logscale::Tuple{Vararg{String}} = (), ODE_options = (AutoTsit5(Rosenbrock23()),))
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    problems = Tuple(create_problem(m.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time)) for person in data)
    indices_θ = [ModelingToolkit.parameter_index(sys, x).idx for x in keys(θ.PK)]
    θ_use = deepcopy(θ)
    partial_transform_to_logscale!(θ_use, logscale = logscale)
    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ)
    negloglikeli = get_negloglikelihood(θ_use, p)
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

function optimise(m::FullModel, data::Tuple; maxiters::Int64 = 10^4, logscale::Tuple{Vararg{String}} = (), inv_hess_CI::Bool = false, bound_abs::Union{Nothing, AbstractFloat} = nothing, lower_upper::Union{Nothing, Tuple{ComponentArray, ComponentArray}} = nothing, 
                objective_fail_hard::Bool = false, objective_warn::Bool = true, solver_optim = LBFGS(linesearch = LineSearches.BackTracking()), ODE_options = (AutoTsit5(Rosenbrock23()),))
    
    #check if either model has random effects
    #if has_random_effects(m.pk_model) || has_random_effects(m.seizure_model)
        #do something to handle them
    names = get_keys_PK(m.pk_model)
    negloglikeli = get_negloglikelihood
    #create ODE problem for each person in data
    sys = create_ode_system(m.pk_model)
    problems = Tuple(create_problem(m.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time)) for person in data)
    #create initial guess
    θ_0 = ComponentArray((PK = m.pk_model.θ, Seizure = m.seizure_model.θ)) 
    #get indices for setting θ
    indices_θ = [ModelingToolkit.parameter_index(sys, x).idx for x in keys(θ_0.PK)]
    #for keys in logscale transform to logscale in θ_0
    partial_transform_to_logscale!(θ_0, logscale = logscale)
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
    if !isnothing(bound_abs)
        θ_0 .= clamp.(θ_0, lb, ub)
    end

    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ, bound_abs = bound_abs, axes_θ = getaxes(θ_0), objective_fail_hard = objective_fail_hard, objective_warn = objective_warn, objective_warned_ref = Ref(false))
    objective = OptimizationFunction(negloglikeli, Optimization.AutoForwardDiff())
    problem = OptimizationProblem(objective, θ_0, p, lb=lb, ub = ub)
    estimate = solve(problem, solver_optim, maxiters = maxiters) 
    #transform parameters back into non logscale
    partial_transform_to_logscale!(estimate.u, logscale = logscale, detransform = true)
    print("Estimate: ", estimate)
    if inv_hess_CI
        CI = inverse_hessian(estimate.u, p, logscale = logscale)
        return estimate, CI
    else
        return estimate
    end
end

function optimise_multistart(m::FullModel, data::Tuple; maxiters::Int64 = 10^4, logscale::Tuple{Vararg{String}} = (), inv_hess_CI::Bool = false, bound_abs::Union{Nothing, AbstractFloat} = nothing, lower_upper::Union{Nothing, Tuple{ComponentArray, ComponentArray}} = nothing,
    objective_fail_hard::Bool = false, objective_warn::Bool = true, multistart::Int = 1, max_threads::Int = multistart, multistart_seed::Union{Nothing, Int} = nothing, multistart_include_initial::Bool = true, multistart_bounds::Union{Nothing, Tuple{AbstractVector, AbstractVector}} = nothing, 
    solver_optim = LBFGS(linesearch = LineSearches.BackTracking()), ODE_options = (AutoTsit5(Rosenbrock23()),))
    
    #check if either model has random effects
    #if has_random_effects(m.pk_model) || has_random_effects(m.seizure_model)
        #do something to handle them
    names = get_keys_PK(m.pk_model)
    #create ODE problem for each person in data
    sys = create_ode_system(m.pk_model)
    problems = Tuple(create_problem(m.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time)) for person in data)
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
    if !isnothing(bound_abs)
        θ_0 .= clamp.(θ_0, lb, ub)
    end

    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ, bound_abs = bound_abs, axes_θ = axes_θ, objective_fail_hard = objective_fail_hard, objective_warn = objective_warn, objective_warned_ref = Ref(false))
    objective = OptimizationFunction(get_negloglikelihood, Optimization.AutoForwardDiff())
    n_starts = max(multistart, 1)
    thread_num = max(max_threads, 1)

    lower = zeros(Float64, d)
    upper = zeros(Float64, d)
    if !isnothing(multistart_bounds)
        lower_raw, upper_raw = multistart_bounds
        if length(lower_raw) != d || length(upper_raw) != d
            error("multistart_bounds must match parameter dimension $d")
        end
        lower .= Float64.(lower_raw)
        upper .= Float64.(upper_raw)
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
        starts[row_idx, :] .= clamp.(Float64.(θ_0_vec), lower, upper)
        row_idx += 1
    end
    n_lhs = n_starts - (multistart_include_initial ? 1 : 0)
    if n_lhs > 0
        rng = isnothing(multistart_seed) ? Random.default_rng() : Random.MersenneTwister(multistart_seed)
        starts[row_idx:end, :] .= latin_hypercube_samples(n_lhs, lower, upper; rng = rng)
    end
    starts_component_vec = [ComponentArray(vec(starts[i, :]), p.axes_θ) for i in 1:n_starts]

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
            problem = OptimizationProblem(objective, starts_component_vec[i], p, lb=lb, ub=ub)
            solutions[i] = solve(problem, solver_optim, maxiters = maxiters)
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
    estimate_u = estimate_raw.u
    #transform parameters back into non logscale
    partial_transform_to_logscale!(estimate_u, logscale = logscale, detransform = true)
    estimate = (u = estimate_u, retcode = estimate_raw.retcode, objective = estimate_raw.objective, raw = estimate_raw, multistart_best_start = best_start_idx, multistart_nstarts = n_starts)
    print("Estimate: ", estimate_raw)
    println("Multistart best start: ", best_start_idx, " Number of Starts: ", n_starts)
    if inv_hess_CI
        CI = inverse_hessian(estimate.u, p, logscale = logscale)
        return estimate, CI
    else
        return estimate
    end
end

#finite_not_forward allows to switch to finite_diff hessian instead of ForwardDiff, often faster but less accurate
function inverse_hessian(θ::ComponentArray, m::FullModel, data::Tuple; confidence::AbstractFloat = 0.95, logscale::Tuple{Vararg{String}} = (), finite_not_forward::Bool = false, ODE_options = (AutoTsit5(Rosenbrock23()),))
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    problems = Tuple(create_problem(m.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time)) for person in data)
    indices_θ = [ModelingToolkit.parameter_index(sys, x).idx for x in keys(θ.PK)]
    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ)
    return inverse_hessian(θ, p, confidence = confidence, logscale = logscale, finite_not_forward=finite_not_forward)
end

function inverse_hessian(θ::ComponentArray, p::NamedTuple; confidence::AbstractFloat = 0.95, logscale::Tuple{Vararg{String}} = (), finite_not_forward::Bool = false)
    #check whether accidentally entered percentage instead of confidence in (0,1)
    if confidence>1
        confidence = confidence/100
        @warn "Presumably you meant a percentage. Your input confidence has been divided by 100"
    end
    f(x) = get_negloglikelihood(x, p)
    θ_use = deepcopy(θ)
    #transform as specified
    partial_transform_to_logscale!(θ_use, logscale = logscale)
    if !(finite_not_forward)
        #This takes very long
        H = ForwardDiff.hessian(f,θ_use)
    else
        H = FiniteDiff.finite_difference_hessian(f, θ_use)
    end
    #Fallback definition of bounds if sth doesn't work
    bounds = [(-Inf, Inf) for i in eachindex(θ_use)]
    try
        H_inv = inv(H)
        #println(H_inv)
        positive_diagonal = true
        for i in eachindex(θ_use)
            positive_diagonal = H_inv[i,i]<0 && positive_diagonal
        end
        if positive_diagonal
            q = quantile(Normal(), (1-(1-confidence)/2))
            #By symmetry other one is just the negative
            #Note q>= 0 since quantile of standard normal positive for >=0.5, ensured for confidence<=1
            bounds = [(θ_use[i] - sqrt(H_inv[i,i])*q, θ_use[i] + sqrt(H_inv[i,i])*q) for i in eachindex(θ_use)]
        else
            @warn "Negative diagonal entry in inverse hessian"
        end
    catch e
        if e isa LinearAlgebra.SingularException
            @warn "Calculated Hessian is singular"
        else
            rethrow(e)
        end
    end
    #now assign intervals to correct keys
    CI = ComponentArray(bounds, getaxes(θ_use))
    #println("CI untransformed: ", CI)
    #transform logscale ones, can't use partial transform since entries are now tuples, have to broadcast
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
    return CI
end

#m determines model parts, n determines number of people, timepoints for measurements
function generate_data(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; timepoints::AbstractVector = 0:14.0:time, wo_treatment::AbstractFloat = 3.0, ODE_options = (AutoTsit5(Rosenbrock23()),))
    population = generate_population(m.population_gen, n)
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    for person in population
        assign_dose!(m.dose_gen, person, names = names, timeframe = time, wo_treatment = wo_treatment)
        sol = generate_measurements!(m.pk_model, sys, person, timepoints = timepoints, endpoint = time, options = ODE_options)
        generate_seizures!(m.seizure_model, sol, person, start = 0.0, day_number = time, names=names)
        #note for time = 10 seizure counts end on day 9 (end on midnight between day 9 and 10)
    end
    return population
end

#for later when want to update doses etc regularly
function generate_data_updating(m::FullModel, n::Int = 10, time::AbstractFloat = 10.0; update_reg::AbstractFloat = time, timepoints::AbstractVector = 0:14.0:time, wo_treatment::AbstractFloat = 3.0, ODE_options = (AutoTsit5(Rosenbrock23()),))
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
            increment = min(time, passed_time + update_reg) - passed_time
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
