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

#basic distribution function for day starting at n, requires sol from chosen PK model
function distribution(m::SeizureBasic, sol, n::AbstractFloat; person::Union{Person, Nothing} = nothing, names::NamedTuple, θ::ComponentArray = m.θ)
    if any(x -> !isfinite(x), (sol(n+1, idxs = names.S)-sol(n,idxs = names.S)))
        return nothing
    end
    intensity = θ.a
    intensity -= θ.b'*(sol(n+1, idxs = names.S)-sol(n,idxs = names.S))
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

#negative binomial distribution function for day starting at n, depends on if seizure on day n-1, requires sol from chosen PK model
function distribution(m::SeizureNegativeBinomial, sol, n::AbstractFloat; person::Person, names::NamedTuple, θ::ComponentArray = m.θ)
    if any(x -> !isfinite(x), (sol(n+1, idxs = names.S)-sol(n,idxs = names.S)))
        return nothing
    end
    o = θ.o
    if o ≤ zero(o) || !isfinite(o)
        return nothing
    end
    seizure_prev_day = (0 < sum([seizure.count for seizure in person.seizure_counts if (n-1 ≤ seizure.time <n)]))
    mean = θ.a + θ.prev*seizure_prev_day
    mean -= θ.b'*(sol(n+1, idxs = names.S)-sol(n,idxs = names.S))
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

#k_n number of seizures on day n
function Seizure_prob_day(m::SeizureModelNonrandom, sol, n::AbstractFloat, k_n::Int64; person::Person, names::NamedTuple, θ::ComponentArray = m.θ)
    distribute = distribution(m,sol,n, person = person, names=names, θ = θ)
    if isnothing(distribute)
        return 0.0
    end
    return pdf(distribute, k_n)
end

function log_Seizure_prob(m::SeizureModelNonrandom, sol, person::Person; θ::ComponentArray = m.θ, names::NamedTuple)
    prob = zero(eltype(θ))
    for i in eachindex(person.seizure_counts) 
        @inbounds time = person.seizure_counts[i].time #get timepoint out of named tuple
        @inbounds count = person.seizure_counts[i].count #get count out of tuple
        log_day_prob = log(Seizure_prob_day(m, sol, time, count, person = person, names = names, θ=θ))
        if !isfinite(log_day_prob)
            return -Inf
        else
            prob += log_day_prob
        end
    end
    return prob
end

function get_seizure_loglikelihood(θ::ComponentArray, m::SeizureModel, sol, person::Person; names::NamedTuple)
    return log_Seizure_prob(m, sol, person, θ=θ, names = names)
end

#3) Implement generation of seizures for discrete, nonrandom models

#generates and appends seizures to person for given number of days start
function generate_seizures!(m::SeizureModelNonrandom, sol, person::Person; start::AbstractFloat = 0.0, day_number::AbstractFloat = 10.0, names::NamedTuple)
    if day_number >=1
    new_seizures = [(time = n, count = rand(distribution(m, sol, n, person = person, names=names))) for n in start:(start+day_number-1)]
    append!(person.seizure_counts, new_seizures)
    end
end