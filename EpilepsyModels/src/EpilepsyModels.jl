module EpilepsyModels

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Optimization
using ForwardDiff
using ComponentArrays
using FiniteDiff

export optimise, generate_data, generate_data_updating, get_negloglikelihood_evaluated, BasicDoses, PolyDoses, PKBasic, PKLEV, 
PKCBZ, BasicPersonGenerator, SeizureBasic, FullModel, PersonGeneratorLEV, BigFourPersonGenerator, PolyDosesRandom,
PKVPA

include("Person Generator.jl")
include("PK Model.jl")
include("Dose Generator.jl")
include("Seizure Model.jl")

struct FullModel{PK<:PKModel, S<:SeizureModel, P<:PersonGenerator, D<:DoseGenerator}
    pk_model::PK
    seizure_model::S
    population_gen::P
    dose_gen::D
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
#p contains m: model, data: tuple, logscale: Tuple{String}, system and problems
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
    system = p.system
    indices = p.indices_θ
    # Keep objective pure: never mutate caller-provided parameter vector.
    θ_use = copy(θ)
    bound_abs = hasproperty(p, :bound_abs) ? p.bound_abs : nothing
    if !isnothing(bound_abs)
        θ_use .= clamp.(θ_use, -bound_abs, bound_abs)
    end
    #for keys in logscale take exponential in θ
    partial_transform_to_logscale!(θ_use, logscale = logscale, detransform = true)
    if any(x -> !isfinite(x), θ_use)
        return Inf
    end
    loglikeli = zero(eltype(θ))
    for i in eachindex(data)
        @inbounds sol = solve_PK(problems[i], system, θ_use.PK, indices_θ = indices, options = options)
        if !(SciMLBase.successful_retcode(sol))
            return Inf
        end
        @inbounds loglikeli += get_PK_loglikelihood(θ_use.PK, data[i], sol=sol)
        @inbounds loglikeli += get_seizure_loglikelihood(θ_use.Seizure, m.seizure_model, sol, data[i], names=names)
    end
    return -loglikeli
end

function get_negloglikelihood_vectorized(θ::AbstractVector, p::NamedTuple)
    try
        if hasproperty(p, :fast_basic) && p.fast_basic
            bound_abs = p.bound_abs
            pk_n = p.pk_n
            # Noise parameter for PK likelihood.
            σ_raw = isnothing(bound_abs) ? θ[p.pk_sigma_idx] : clamp(θ[p.pk_sigma_idx], -bound_abs, bound_abs)
            σ = p.pk_log_mask[p.pk_sigma_idx] ? exp(σ_raw) : σ_raw
            if !(isfinite(σ) && σ > zero(σ))
                return Inf
            end

            a_idx = pk_n + p.seiz_a_idx
            a_raw = isnothing(bound_abs) ? θ[a_idx] : clamp(θ[a_idx], -bound_abs, bound_abs)
            a = p.seiz_log_mask[p.seiz_a_idx] ? exp(a_raw) : a_raw
            if !isfinite(a)
                return Inf
            end

            if length(p.seiz_b_idx) != length(p.names.S)
                return Inf
            end

            loglikeli = zero(eltype(θ))
            log2pi_over_2 = oftype(loglikeli, log(2π) / 2)
            θ_solve = ComponentArray(copy(θ), p.axes_θ)
            partial_transform_to_logscale!(θ_solve, logscale = p.logscale, detransform = true)
            @inbounds for i in eachindex(p.data)
                sol = solve_PK(p.problems[i], p.system, θ_solve.PK, indices_θ = p.indices_θ, options = p.options)
                if !(SciMLBase.successful_retcode(sol))
                    return Inf
                end
                # PK log-likelihood.
                for measure in p.data[i].measurements
                    μ = sol(measure.timepoint, idxs = measure.state[2])
                    if !isfinite(μ)
                        return Inf
                    end
                    z = (measure.measurement - μ) / σ
                    loglikeli += -(log(σ) + log2pi_over_2 + (z * z) / 2)
                end
                # Seizure log-likelihood for SeizureBasic.
                for sc in p.data[i].seizure_counts
                    intensity = a
                    for j in eachindex(p.names.S)
                        state_name = p.names.S[j]
                        exposure = sol(sc.time + 1, idxs = state_name) - sol(sc.time, idxs = state_name)
                        if !isfinite(exposure)
                            return Inf
                        end
                        idx_rel = p.seiz_b_idx[j]
                        idx = pk_n + idx_rel
                        b_raw = isnothing(bound_abs) ? θ[idx] : clamp(θ[idx], -bound_abs, bound_abs)
                        b = p.seiz_log_mask[idx_rel] ? exp(b_raw) : b_raw
                        if !isfinite(b)
                            return Inf
                        end
                        intensity += b * exposure
                    end
                    intensity = max(zero(intensity), intensity)
                    if !isfinite(intensity)
                        return Inf
                    end
                    if intensity == zero(intensity)
                        if sc.count == 0
                            # log P(K=0 | λ=0) = 0
                            loglikeli += zero(loglikeli)
                        else
                            return Inf
                        end
                    else
                        logfact = zero(loglikeli)
                        for r in 2:sc.count
                            logfact += log(r)
                        end
                        loglikeli += sc.count * log(intensity) - intensity - logfact
                    end
                end
            end
            return -loglikeli
        end

        θ_struct = ComponentArray(copy(θ), p.axes_θ)
        out = get_negloglikelihood(θ_struct, p)
        return isfinite(out) ? out : Inf
    catch e
        fail_hard = hasproperty(p, :objective_fail_hard) ? p.objective_fail_hard : false
        if fail_hard
            rethrow(e)
        end
        if hasproperty(p, :objective_warned_ref) && hasproperty(p, :objective_warn)
            if p.objective_warn && !p.objective_warned_ref[]
                p.objective_warned_ref[] = true
                @warn "Objective evaluation failed; returning Inf for this candidate." #exception=(e, catch_backtrace())
            end
        end
        return Inf
    end
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

#Latin hypercube design in box constraints [lower, upper].
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

function optimise(m::FullModel, data::Tuple; maxiters::Int64 = 10^4, logscale::Tuple{Vararg{String}} = (), inv_hess_CI::Bool = false, bound_abs::Union{Nothing, AbstractFloat} = 10.0, objective_fail_hard::Bool = false, objective_warn::Bool = true, multistart::Int = 1, multistart_seed::Union{Nothing, Int} = nothing, multistart_include_initial::Bool = true, multistart_bounds::Union{Nothing, Tuple{AbstractVector, AbstractVector}} = nothing, solver_optim = LBFGS(linesearch = LineSearches.BackTracking()), ODE_options = (AutoTsit5(Rosenbrock23()),))
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
    if !isnothing(bound_abs)
        θ_0 .= clamp.(θ_0, -bound_abs, bound_abs)
    end

    pk_log_mask = falses(length(θ_0.PK))
    seiz_log_mask = falses(length(θ_0.Seizure))
    for label in logscale
        if label in labels(θ_0.PK) || Symbol(label) in keys(θ_0.PK)
            for idx in label2index(θ_0.PK, label)
                pk_log_mask[idx] = true
            end
        end
        if label in labels(θ_0.Seizure) || Symbol(label) in keys(θ_0.Seizure)
            for idx in label2index(θ_0.Seizure, label)
                seiz_log_mask[idx] = true
            end
        end
    end

    pk_sigma_idx = label2index(θ_0.PK, "σ")[1]
    seiz_a_idx = label2index(θ_0.Seizure, "a")[1]
    seiz_b_idx = collect(label2index(θ_0.Seizure, "b"))
    fast_basic = m.seizure_model isa SeizureBasic

    axes_θ = getaxes(θ_0)
    θ_0_vec = collect(θ_0)
    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ, bound_abs = bound_abs, axes_θ = axes_θ, objective_fail_hard = objective_fail_hard, objective_warn = objective_warn, objective_warned_ref = Ref(false), fast_basic = fast_basic, pk_n = length(θ_0.PK), pk_log_mask = pk_log_mask, seiz_log_mask = seiz_log_mask, pk_sigma_idx = pk_sigma_idx, seiz_a_idx = seiz_a_idx, seiz_b_idx = seiz_b_idx)
    objective = OptimizationFunction(get_negloglikelihood_vectorized, Optimization.AutoForwardDiff())
    n_starts = max(multistart, 1)
    d = length(θ_0_vec)

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
        lower .= -Float64(bound_abs)
        upper .= Float64(bound_abs)
    else
        #Fallback finite box around initial point in unconstrained mode.
        lower .= Float64.(θ_0_vec) .- 2.0
        upper .= Float64.(θ_0_vec) .+ 2.0
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

    best_raw_any = nothing
    best_start_any = 1
    best_raw_finite = nothing
    best_obj_finite = Inf
    best_start_finite = 1
    best_raw_success_finite = nothing
    best_obj_success_finite = Inf
    best_start_success_finite = 1
    for s_idx in 1:n_starts
        problem = OptimizationProblem(objective, vec(starts[s_idx, :]), p)
        estimate_raw_i = solve(problem, solver_optim, maxiters = maxiters)
        if isnothing(best_raw_any)
            best_raw_any = estimate_raw_i
            best_start_any = s_idx
        end
        finite_i = isfinite(estimate_raw_i.objective)
        if finite_i
            obj_i = estimate_raw_i.objective
            if isnothing(best_raw_finite) || obj_i < best_obj_finite
                best_raw_finite = estimate_raw_i
                best_obj_finite = obj_i
                best_start_finite = s_idx
            end
            if SciMLBase.successful_retcode(estimate_raw_i.retcode) && (isnothing(best_raw_success_finite) || obj_i < best_obj_success_finite)
                best_raw_success_finite = estimate_raw_i
                best_obj_success_finite = obj_i
                best_start_success_finite = s_idx
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
    estimate_u = ComponentArray(copy(estimate_raw.u), axes_θ)
    if !isnothing(bound_abs)
        estimate_u .= clamp.(estimate_u, -bound_abs, bound_abs)
    end
    #transform parameters back into non logscale
    partial_transform_to_logscale!(estimate_u, logscale = logscale, detransform = true)
    estimate = (u = estimate_u, retcode = estimate_raw.retcode, objective = estimate_raw.objective, raw = estimate_raw, multistart_best_start = best_start_idx, multistart_nstarts = n_starts)
    print("Estimate: ", estimate)
    if inv_hess_CI
        CI = inverse_hessian(estimate.u, p, logscale = logscale)
        return estimate, CI
    else
        return estimate
    end
end

function inverse_hessian(θ::ComponentArray, m::FullModel, data::Tuple; confidence::AbstractFloat = 0.95, logscale::Tuple{Vararg{String}} = (), ODE_options = (AutoTsit5(Rosenbrock23()),))
    names = get_keys_PK(m.pk_model)
    sys = create_ode_system(m.pk_model)
    problems = Tuple(create_problem(m.pk_model, sys, person=person, endpoint = max(person.measurements[end].timepoint, person.seizure_counts[end].time)) for person in data)
    indices_θ = [ModelingToolkit.parameter_index(sys, x).idx for x in keys(θ.PK)]
    p = (m = m, data = data, logscale = logscale, options = ODE_options, names=names, problems = problems, system = sys, indices_θ = indices_θ)
    return inverse_hessian(θ, p, confidence = confidence, logscale = logscale)
end

function inverse_hessian(θ::ComponentArray, p::NamedTuple; confidence::AbstractFloat = 0.95, logscale::Tuple{Vararg{String}} = ())
    #check whether accidentally entered percentage instead of confidence in (0,1)
    if confidence>1
        confidence = confidence/100
        @warn "Presumably you meant a percentage. Your input confidence has been divided by 100"
    end
    f(x) = get_negloglikelihood(x, p)
    θ_use = deepcopy(θ)
    #transform as specified
    partial_transform_to_logscale!(θ_use, logscale = logscale)
    println("Starting hessian forwarddiff")
    #This takes very long
    H = ForwardDiff.hessian(f,θ_use)
    println("H forwarddiff= ", H)
    println("Starting hessian finitediff")
    H_finite = FiniteDiff.finite_difference_hessian(f, θ_use)
    println("Starting inverse")
    H_inv = inv(H)
    H_inv_finite = inv(H_finite)
    println("H_inv finitediff = ")
    println(H_inv_finite)
    println("H_inv forwarddiff = ")
    println(H_inv)
    for i in eachindex(θ)
        if H_inv[i,i]<0
            error("negative diagonal entry in inverse hessian")
        end
    end
    q = quantile(Normal(), (1-(1-confidence)/2))
    #By symmetry other one is just the negative
    #Note q>= 0 since quantile of standard normal positive for >=0.5, ensured for confidence<=1
    bounds = [[θ_use[i] - sqrt(H_inv[i,i])*q, θ_use[i] + sqrt(H_inv[i,i])*q] for i in eachindex(θ_use)]
    #now assign correct keys, transform logscale ones
    #this works but single components are now one entry vectors
    CI = ComponentArray(PK = ComponentArray((; [(key,bounds[label2index(θ, "PK.$(key)")]) for key in keys(θ.PK)]...)), 
                            Seizure = ComponentArray((; [(key,bounds[label2index(θ, "Seizure.$(key)")]) for key in keys(θ.Seizure)]...)))
    partial_transform_to_logscale!(CI, logscale = logscale, detransform = true)
    #alternatively could do for i in eachindex(θ) θ[i] = bounds[i][1] end and same for two, then put together, can't assign directly since different types
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
