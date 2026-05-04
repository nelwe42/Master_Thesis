using ModelingToolkit
using Distributions
using Random
using Parameters
using ComponentArrays
using StaticArrays

#Overtype of Seizure Models that will go into full model
abstract type SeizureModel end

#To distinguish if possibly decide to make time continuous models later
abstract type SeizureModelDiscrete <: SeizureModel end

#later for checking if random effects need to be handled in inference
abstract type SeizureModelNonrandom <: SeizureModelDiscrete end
#For this need some getter for which are random effects?

#Every model specification should have: ComponentArray of parameters, list of keys of required covariates
#Every model should have function returning distribution given day/further information

#Within discrete/continuous and (non)random returning seizure probability, likelihoods and 
#generating data can be handled once

#1)Specific model instances with their intensities

@with_kw struct SeizureBasic{T<:ComponentArray, T2<:Tuple} <: SeizureModelNonrandom
    θ::T=ComponentArray((a = 2.0, b = SA[0.0])) #a base rate, b coefficient of drug 
    cov::T2 = () #no covariates required
end

#Outer Constructor to make default for N drugs
function SeizureBasic(N::Int64)
    obj = SeizureBasic(θ = ComponentArray((a = 2.0, b = SA[0 for i in 1:N])))
    return obj
end

#basic distribution function for record_interval in days starting at n, requires sol from chosen PK model
function distribution(m::SeizureBasic, sol, n::AbstractFloat; person::Union{Person, Nothing} = nothing, record_interval::AbstractFloat = 1.0, names::NamedTuple, θ::ComponentArray = m.θ)
    if any(x -> !isfinite(x), (sol(n+1, idxs = names.S)-sol(n,idxs = names.S)))
        return nothing
    end
    intensity = θ.a*record_interval
    intensity -= θ.b'*(sol(n+record_interval, idxs = names.S)-sol(n,idxs = names.S))
    if !isfinite(intensity)
        return nothing
    end
    distribution = Poisson(max(0,intensity))
    #on day n natural number beginning with 0 are exposed to drug from time n to n+1
    #day 0 ist interval (0,1], day named after first number
    return distribution
end

@with_kw struct SeizureNegativeBinomial{T<:ComponentArray, T2<:Tuple} <: SeizureModelNonrandom
    θ::T=ComponentArray((a = log(2.0), o = 0.01, prev = 0.0, b = SA[0.0])) #a base rate, prev impact of previous day, o overdispersion, b coefficient of drug 
    cov::T2 = (:seizure_prev_day,) #depends on if seizure occured on previous day
end

#Outer Constructor to make default for N drugs
function SeizureNegativeBinomial(N::Int64)
    obj = SeizureNegativeBinomial(θ = ComponentArray((a = 2.0, o = 0.01, prev = 0.0, b = SA[0 for i in 1:N])))
    return obj
end

#negative binomial distribution function for record_interval starting at n, depends on if seizure on previous interval, requires sol from chosen PK model
function distribution(m::SeizureNegativeBinomial, sol, n::AbstractFloat; person::Person, record_interval::AbstractFloat = 1.0, names::NamedTuple, θ::ComponentArray = m.θ)
    if any(x -> !isfinite(x), (sol(n+1, idxs = names.S)-sol(n,idxs = names.S)))
        return nothing
    end
    o = θ.o
    if o ≤ zero(o) || !isfinite(o)
        return nothing
    end
    seizure_prev_day = (0 < sum([seizure.count for seizure in person.seizure_counts if (n-record_interval ≤ seizure.time <n)]))
    mean = θ.a*record_interval + θ.prev*seizure_prev_day*(1/record_interval)
    mean -= θ.b'*(sol(n+record_interval, idxs = names.S)-sol(n,idxs = names.S))
    if !isfinite(mean)
        return nothing
    end
    mean = exp(mean)
    #Transform from representation with mean to with success probability
    p = o/(mean+o)
    if !(zero(p) < p ≤ one(p))
        return nothing
    end
    distribution = NegativeBinomial(o,p)
    #on day n natural number beginning with 0 are exposed to drug from time n to n+1
    #day 0 ist interval (0,1], day named after first number
    return distribution
end

#2) Implement Seizure Probabilities, Likelihoods and Data Generators for discrete, nonrandom

#k_n number of seizures (or presence/absence) on day n
function Seizure_prob_interval(m::SeizureModelNonrandom, sol, n::AbstractFloat, k_n::Union{Int64, Bool}, record_interval::AbstractFloat; person::Person, names::NamedTuple, θ::ComponentArray = m.θ)
    distribute = distribution(m,sol,n, person = person, record_interval = record_interval, names=names, θ = θ)
    if isnothing(distribute)
        return 0.0
    end
    if (k_n isa Bool) && (eltype(distribute) != Bool)
        prob_false = pdf(distribute, 0)
        if k_n 
            return 1-prob_false
        else
            return prob_false
        end
    else
        return pdf(distribute, k_n)
    end
end

function log_Seizure_prob(m::SeizureModelNonrandom, sol, person::Person; θ::ComponentArray = m.θ, names::NamedTuple)
    prob = zero(eltype(θ))
    for i in eachindex(person.seizure_counts) 
        @inbounds time = person.seizure_counts[i].time #get time interval out of named tuple
        @inbounds count = person.seizure_counts[i].count #get count out of tuple
        start = time[1]
        interval = time[2] - time[1]
        if interval < 0
            error("Seizure time interval bounds in wrong order")
        end
        log_interval_prob = log(Seizure_prob_interval(m, sol, start, count, interval, person = person, names = names, θ=θ))
        if !isfinite(log_interval_prob)
            return -Inf
        else
            prob += log_interval_prob
        end
    end
    return prob
end

function get_seizure_loglikelihood(θ::ComponentArray, m::SeizureModel, sol, person::Person; names::NamedTuple)
    return log_Seizure_prob(m, sol, person, θ=θ, names = names)
end

#3) Implement generation of seizures for discrete, nonrandom models

#generates and appends seizures to person for given number of days start
function generate_seizures!(m::SeizureModelNonrandom, sol, person::Person; timepoints::AbstractVector=0.0:1.0:10.0, just_Bool::Bool = false, names::NamedTuple)
    if !(just_Bool)
        new_seizures = [(time = (timepoints[i], timepoints[i+1]), count = rand(distribution(m, sol, timepoints[i], record_interval = timepoints[i+1]-timepoints[i], person = person, names=names))) for i in 1:(length(timepoints)-1)]
    else
        new_seizures = [(time = (timepoints[i], timepoints[i+1]), count = (rand(distribution(m, sol, timepoints[i], record_interval = timepoints[i+1]-timepoints[i], person = person, names=names))>0)) for i in 1:(length(timepoints)-1)]
    end
    append!(person.seizure_counts, new_seizures)
end

#4) Functions for visualisation

#Plots fit for person i up to time given solutions from PK model (if they should be plotted)
function plot_fit(mod::SeizureModel, data::Tuple; estimate_param::Union{ComponentArray, Nothing} = nothing, sols_true::Union{AbstractVector, Nothing} = nothing, sols_estimated::Union{AbstractVector, Nothing} = nothing, 
    names::NamedTuple, individuals::AbstractVector = [1], time::Union{Tuple{Union{Int, AbstractFloat}, Union{Int, AbstractFloat}}, AbstractFloat, Int} = 10, display_plot::Bool = true)
    
    output = Plots.Plot[]
    if !isnothing(sols_true)
        endpoint = sols_true[1].t[end]
    elseif !isnothing(sols_estimated)
        endpoint = sols_estimated[1].t[end]
    else
        endpoint = max(data[1].measurements[end].timepoint, data[1].seizure_counts[end].time[2])
    end
    if time isa Number
        time = (0,time)
    end
    if time[1]<0 || time[2] > endpoint || time[1] > time[2]
        error("Incorrectly defined time window for seizure plotting")
    end
    #Cut down time to full days
    time = Int.(floor.(time))
    if isnothing(sols_true)
        sols = nothing
    else
        sols = sols_true[individuals]
    end
    if !isnothing(sols) && any(.!(SciMLBase.successful_retcode.(sols)))
        @warn "Unsuccessful ODE solve in true parameters, true parameters will be ignored for plotting"
        sols = nothing
    end
    if isnothing(sols_estimated)
        sols2 = nothing
    else
        sols2 = sols_estimated[individuals]
    end
    if !isnothing(estimate_param) && isnothing(sols2)
        error("Estimate solutions are missing")
    end
    if !isnothing(sols2) && any(.!(SciMLBase.successful_retcode.(sols2)))
        @warn "Unsuccessful ODE solve in estimated parameters, estimated parameters will be ignored for plotting"
        estimate_param = nothing
    end
    if !isnothing(estimate_param)
        seizure_model_estimate = typeof(mod).name.wrapper(θ = estimate_param)
    end
    for i in individuals
        pl2 = plot(xlabel = "day", ylabel = "Seizure Probability", title = "Seizure probabilities for person $(i) for day $(time[1]) to $(time[2])")
        if !isnothing(sols)
            distribution_true = [distribution(mod, sols[i], Float64(n), person=data[i], names=names) for n in time[1]:time[2]]
            if any(isnothing.(distribution_true))
                @warn "Distribution for true parameters is not well-defined"
                sols = nothing
            end
        end
        if !isnothing(estimate_param)
            distribution_estimate = [distribution(seizure_model_estimate, sols2[i], Float64(n), person=data[i], names=names) for n in time[1]:time[2]]
            if any(isnothing.(distribution_estimate))
                @warn "Distribution for estimate parameters is not well-defined"
                estimate_param = nothing
            end
        end
        #Plot distributions, first day separate so label only once, only on separate sides if not both plotted
        if !isnothing(estimate_param) && !isnothing(sols)
            violin!(["$(time[1])"], rand(distribution_true[1],1000), side = :left, label = "true", colour = :dodgerblue)
            violin!(["$(time[1])"], rand(distribution_estimate[1],1000), side = :right, label = "estimate", colour = :firebrick2)
            for day in (time[1]+1):time[2]
                violin!(["$(day)"], rand(distribution_true[day-time[1]+1],1000), side = :left, label = "", colour = :dodgerblue)
                violin!(["$(day)"], rand(distribution_estimate[day-time[1]+1],1000), side = :right, label = "", colour = :firebrick2)
            end
        elseif !isnothing(estimate_param)
            violin!(["$(time[1])"], rand(distribution_estimate[1],1000), label = "estimate", colour = :firebrick2)
            for day in (time[1]+1):time[2]
                violin!(["$(day)"], rand(distribution_estimate[day-time[1]+1],1000), label = "", colour = :firebrick2)
            end
        elseif !isnothing(sols)
            violin!(["$(time[1])"], rand(distribution_true[1],1000), label = "true", colour = :dodgerblue)
            for day in (time[1]+1):time[2]
                violin!(["$(day)"], rand(distribution_true[day-time[1]+1],1000), label = "", colour = :dodgerblue)
                end
        end
        #Add where data is
        boxplot!(["$(time[1])"], [seizure.count for seizure in data[i].seizure_counts if (time[1] ≤ seizure.time[1] < (time[1]+1))], label = "Data values", colour = :grey, linewidth = 3)
        for day in (time[1]+1):time[2]
            boxplot!(["$(day)"], [seizure.count for seizure in data[i].seizure_counts if (day ≤ seizure.time[1] < day+1)], label = "", colour = :grey, linewidth = 3)
        end
        #add to output
        append!(output, [pl2])
        if display_plot
            display(pl2)
        end
    end
    return output
end