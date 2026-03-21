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
#Every model should have function returning intensity (if is a Poisson based model, likely most will be)

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

#basic intensity function for day starting at n, requires sol from chosen PK model
function intensity(m::SeizureBasic, sol, n::AbstractFloat; covariates = nothing, names::NamedTuple, θ::ComponentArray = m.θ)
    if any(x -> !isfinite(x), (sol(n+1, idxs = names.S)-sol(n,idxs = names.S)))
        return Inf
    end
    intensity = θ.a
    intensity -= θ.b'*(sol(n+1, idxs = names.S)-sol(n,idxs = names.S))
    #on day n natural number beginning with 0 are exposed to drug from time n to n+1
    #day 0 ist interval (0,1], day named after first number
    return max(0,intensity)
end

#2) Implement Seizure Probabilities, Likelihoods and Data Generators for discrete, nonrandom

#k_n number of seizures on day n
function Seizure_prob_day(m::SeizureModelNonrandom, sol, n::AbstractFloat, k_n::Int64; covariates = nothing, names::NamedTuple, θ::ComponentArray = m.θ)
    lambda = intensity(m,sol,n, covariates = covariates, names=names, θ = θ)
    if !isfinite(lambda)
        return 0.0
    end
    return ((lambda^k_n)/factorial(k_n))*exp(-lambda)
end

function log_Seizure_prob(m::SeizureModelNonrandom, sol, person::Person; θ::ComponentArray = m.θ, names::NamedTuple)
    prob = zero(eltype(θ))
    for i in eachindex(person.seizure_counts) 
        @inbounds time = person.seizure_counts[i].time #get timepoint out of named tuple
        @inbounds count = person.seizure_counts[i].count #get count out of tuple
        cov = NamedTuple{m.cov}(person.covariates) #create cov via person covariates and keys
        log_day_prob = log(Seizure_prob_day(m, sol, time, count, covariates = cov, names = names, θ=θ))
        if !isfinite(log_day_prob)
            return Inf
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
    cov = NamedTuple{m.cov}(person.covariates)
    new_seizures = [(time = n, count = rand(Poisson(intensity(m, sol, n, covariates=cov, names=names)))) for n in start:(start+day_number-1)]
    append!(person.seizure_counts, new_seizures)
    end
end