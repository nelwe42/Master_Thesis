using Random
using Distributions
using Parameters

#set seed
Random.seed!(42)

@with_kw struct Person{T<:NamedTuple}
    covariates::T = NamedTuple() #named tuple of covariates
    dosing::AbstractVector = Vector{@NamedTuple{t::AbstractFloat, dose::AbstractFloat, state::Symbol}}() #vector of NamedTuples of the form t, dose, state
    seizure_counts::AbstractVector = Vector{@NamedTuple{time::AbstractFloat, count::AbstractFloat}}() #vector of NamedTuples of the form time, count
    measurements::AbstractVector = Vector{@NamedTuple{timepoint::AbstractFloat, measurement::AbstractFloat}}() #vector of NamedTuples of the form timepoint, measurement
    #later make attribute with individual values of random effects
end

abstract type PersonGenerator end

struct BasicPersonGenerator <: PersonGenerator end

function generate_population(m::BasicPersonGenerator, n::Int = 10)
    population = [Person() for i in 1:n]
    return population
end

