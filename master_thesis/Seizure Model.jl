using ModelingToolkit
using Distributions
using Random
using Parameters

#set seed
Random.seed!(42)

#Overtype of Seizure Models that will go into full model
abstract type Seizure_Model end

#To distinguish if possibly decide to make time continuous models later
abstract type Seizure_Model_discrete <: Seizure_Model end

#later for checking if random effects need to be handled in inference
abstract type Seizure_Model_nonrandom <: Seizure_Model_discrete end
#For this need some getter for which are random effects?

#Every model specification should have: Named tuple of parameters, list of keys of required covariates
#Every model should have function returning intensity

#Within discrete/continuous and (non)random returning seizure probability, likelihoods and 
#generating data can be handled once

#1)Specific model instances with their intensities

@with_kw struct Seizure_Basic{T<:NamedTuple, T2<:Tuple} <: Seizure_Model_nonrandom
    θ::T=(a = 0.0, b = 0.0) #a base rate, b coefficient of drug (how to handle more later?)
    cov::T2 = () #no covariates required
end

#function intensity(mod::Seizure_Basic, n::Int; here take PK system/solution object)
function intensity(m::Seizure_Basic, sol, n::AbstractFloat; covariates = nothing)
    intensity = m.θ.a
    intensity = intensity + m.θ.b*(sol(n, idxs = S)-sol(n-1,idxs = S))
    #on day n natural number beginning with 1 are exposed to drug from time n-1 to n
    #day 1 ist interval (0,1]
    return intensity
end

#2) Implement Seizure Probabilities, Likelihoods and Data Generators for discrete, nonrandom

#k_n number of seizures on day n
function Seizure_prob_day(m::Seizure_Model_nonrandom, sol, n::AbstractFloat, k_n::AbstractFloat; covariates = nothing)
    lambda = intensity(m,sol,n, covariates = covariates)
    return (lambda^k_n/factorial(k_n))*exp(-lambda)
end

function Seizure_prob(m::Seizure_Model_nonrandom, sol, person::Person)
    prob = 1
    for i in eachindex(person.seizure_counts) 
        time = person.seizure_counts[i].time #get timepoint out of named tuple
        count = person.seizure_counts[i].count #get count out of tuple
        cov = NamedTuple{m.cov}(person.covariates) #create cov via person covariates and keys
        prob = prob * Seizure_prob_day(m::Seizure_Model_nonrandom, sol, time, count, covariates = cov)
    end
    return prob
end

function get_seizure_loglikelihood(θ::NamedTuple, m::Seizure_Model, sol, person::Person)
    m_set = typeof(m)(θ = θ) 
    likeli = Seizure_prob(m_set, sol, person)
    return log(likeli)
end

#3) Implement generation of seizures for discrete, nonrandom models

#generates and appends seizures to person for given number of days start
function generate_seizures!(m::Seizure_Model_nonrandom, sol, person::Person, start::AbstractFloat; day_number::AbstractFloat = 10.0)
    cov = NamedTuple{m.cov}(person.covariates)
    new_seizures = [rand(Poisson(intensity(m, sol, n, covariates=cov))) for n in start:(start+day_number)]
    append!(person.seizure_counts, new_seizures)
end