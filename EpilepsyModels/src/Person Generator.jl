using Random
using Distributions
using Parameters

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

#Under construction, should create covariates for all big 4 drugs LEV, CBZ, VPA, LTG
@with_kw struct BigFourPersonGenerator{T<:AbstractFloat} <: PersonGenerator 
    #probability of previous CBZ therapy and kidney disease can be adjusted 
    prob_kidney_disease::T = 0.1
    prob_prev_CBZ::T = 0.0
end

function generate_population(m::BigFourPersonGenerator, n::Int = 10)
    #draw height from normal distribution
    #draw first BMI, weight = BMI*height(in m)^2
    #average BMI Germany was 26 in 2021, 45,3% in 18.5 to 25, 35.9 in 25 to 30
    #draw kindey_disease as binomial, potentially later dependent on weight
    #draw heights first because weights dependent on them
    #gender equally distributed, 1 for female, 0 for male
    heights = Tuple(rand(Normal(170,7)) for i in 1:n)
    @inbounds population = Tuple(Person(covariates = (height = heights[i], weight = rand(Normal(26.0, 3.5))*(heights[i]/100)^2, 
                                kidney_disease = Float64(rand(Bernoulli(m.prob_kidney_disease))), 
                                prev_CBZ = Float64(rand(Bernoulli(m.prob_prev_CBZ))), gender = Float64(rand(Bernoulli(0.5))))) for i in 1:n)
    return population
end