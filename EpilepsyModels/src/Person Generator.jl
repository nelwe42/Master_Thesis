using Random
using Distributions
using Parameters

#set seed
Random.seed!(42)

@with_kw struct Person{T<:NamedTuple, D<:AbstractVector{<:NamedTuple}, S<:AbstractVector{<:NamedTuple}, M<:AbstractVector{<:NamedTuple}}
    covariates::T = NamedTuple() #named tuple of covariates
    dosing::D = Vector{@NamedTuple{t::AbstractFloat, dose::AbstractFloat, state::Symbol}}() #vector of NamedTuples of the form t, dose, state
    seizure_counts::S = Vector{@NamedTuple{time::AbstractFloat, count::Int64}}() #vector of NamedTuples of the form time, count
    measurements::M = Vector{@NamedTuple{timepoint::AbstractFloat, measurement::AbstractFloat, state::Tuple{Symbol, Symbol}}}() #vector of NamedTuples of the form timepoint, measurement
    #later make attribute with individual values of random effects
end

abstract type PersonGenerator end

struct BasicPersonGenerator <: PersonGenerator end

function generate_population(m::BasicPersonGenerator, n::Int = 10)
    population = Tuple(Person() for i in 1:n)
    return population
end

struct PersonGeneratorLEV <: PersonGenerator end

function generate_population(m::PersonGeneratorLEV, n::Int = 10)
    #draw height from normal distribution
    #draw first BMI, weight = BMI*height(in m)^2
    #average BMI Germany was 26 in 2021, 45,3% in 18.5 to 25, 35.9 in 25 to 30
    #draw kindey_disease as binomial, potentially later dependent on weight
    #draw heights first because weights dependent on them
    heights = Tuple(rand(Normal(170,7)) for i in 1:n)
    @inbounds population = Tuple(Person(covariates = (height = heights[i], weight = rand(Normal(26.0, 3.5))*(heights[i]/100)^2, kidney_disease = Float64(rand(Bernoulli(0.1))))) for i in 1:n)
    return population
end

