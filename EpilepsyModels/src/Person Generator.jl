using Random
using Distributions
using Parameters

#For continuous models, seizure count of 0 at a time means censored
#generally time is float for cox, if censored in middle may have tuple of times between which next even occurs and count false
@with_kw struct Person{T<:NamedTuple, D<:AbstractVector{<:NamedTuple}, S<:AbstractVector{<:NamedTuple}, M<:AbstractVector{<:NamedTuple}, R<:AbstractVector}
    covariates::T = NamedTuple() #named tuple of covariates
    #Float64 not AbstractFloat: these hold generated data, never dual numbers, and an abstract field type
    #boxes every entry and makes every function reading them type unstable
    dosing::D = Vector{@NamedTuple{t::Float64, dose::Float64, state::Symbol}}() #vector of NamedTuples of the form t, dose, state
    seizure_counts::S = Vector{@NamedTuple{time::Union{Tuple{Float64, Float64}, Float64}, count::Union{Int64, Bool}}}() #vector of NamedTuples of the form time, count
    measurements::M = Vector{@NamedTuple{timepoint::Float64, measurement::Float64, state::Tuple{Symbol, Symbol}}}()
    #vector of NamedTuples of the form timepoint, measurement, state being measured, both obs and corresponding s
    #later make attribute with individual values of random effects
    random_effects::R = Vector()
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
    #draw first BMI, weight = BMI*height(in m)^2 to roughly capture weight height relationship
    #average BMI Germany was 26 in 2021, 45,3% in 18.5 to 25, 35.9 in 25 to 30
    #draw kindey_disease as binomial, potentially later dependent on weight
    #draw heights first because weights dependent on them
    heights = Tuple(rand(Normal(170,7)) for i in 1:n)
    @inbounds population = Tuple(Person(covariates = (height = heights[i], weight = rand(Normal(26.0, 3.5))*(heights[i]/100)^2, kidney_disease = Float64(rand(Bernoulli(0.1))))) for i in 1:n)
    return population
end

#Under construction, should create covariates for all big 4 drugs LEV, CBZ, VPA, LTG
@with_kw struct BigFourPersonGenerator{T<:AbstractFloat, T2<:Distribution} <: PersonGenerator 
    #probability of previous CBZ therapy and kidney disease can be adjusted 
    prob_kidney_disease::T = 0.1
    prob_prev_CBZ::T = 0.0
    prob_smoking::T = 0.189 #ratio of smokers Germany 2021 according to Mikrozensus, ignoring differences in sex, age, Bundesland
    creatinine_distr = ((101, 30.5/1.96), (86.9, 25.8/1.96)) #info on creatinine distr for each gender
    prob_focal::T = 0.6 #probability of having focal seizures
    age_distr::T2 = MixtureModel([Binomial(90, 15.0/90), Binomial(90, 65/90)], Categorical(0.35, 0.65))
end

function generate_population(m::BigFourPersonGenerator, n::Int = 10)
    #draw height from normal distribution
    #draw first BMI, weight = BMI*height(in m)^2
    #average BMI Germany was 26 in 2021, 45,3% in 18.5 to 25, 35.9 in 25 to 30
    #draw kindey_disease as binomial, potentially later dependent on weight
    #draw heights first because weights dependent on them
    #gender equally distributed, 1 for female, 0 for male
    heights = Tuple(rand(Normal(170,7)) for i in 1:n)
    genders = Tuple(Float64(rand(Bernoulli(0.5))) for i in 1:n)
    @inbounds population = Tuple(Person(covariates = (height = heights[i], weight = rand(Normal(26.0, 3.5))*(heights[i]/100)^2, 
                                kidney_disease = Float64(rand(Bernoulli(m.prob_kidney_disease))), 
                                CLCr = rand(Normal(m.creatinine_distr[genders[i]+1]...)),
                                prev_CBZ = Float64(rand(Bernoulli(m.prob_prev_CBZ))), 
                                smoking = Float64(rand(Bernoulli(m.prob_smoking))), 
                                seizure_type = Float64(rand(Bernoulli(m.prob_focal))),
                                age = rand(m.age_distr),
                                gender = genders[i])) for i in 1:n)
    return population
end