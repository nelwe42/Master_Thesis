using ModelingToolkit
using Distributions
using Random
using Parameters
using ComponentArrays
using StaticArrays
using Combinatorics

#Overtype of Seizure Models that will go into full model
abstract type SeizureModel end

#To distinguish if possibly decide to make time continuous models later
abstract type SeizureModelDiscrete <: SeizureModel end

#later for checking if random effects need to be handled in inference
abstract type SeizureModelNonrandom <: SeizureModelDiscrete end
#For this need some getter for which are random effects?

#Every model specification should have: ComponentArray of parameters, list of keys of required covariates
#specify timeframe model supports, what kind of autocorrelation it needs
#models need attribute timeframe = (general_timeframe = yes/no, inherent_timeframe = length in days e.g. 1.0)
#models need bool attribute autocorrelation, length if yes, i.e. autocorrelation = (yes/no, timeframe)
#if have autocorrelation distribution takes extra argument seizures, if general timeframe takes extra argument record_interval
#Every model should have function returning distribution given day/further information

#Within discrete/continuous and (non)random returning seizure probability, likelihoods and 
#generating data can be handled once

#1)Specific model instances with their intensities

@with_kw struct SeizureBasic{T<:ComponentArray, T2<:Tuple} <: SeizureModelNonrandom
    θ::T=ComponentArray((a = 2.0, b = SA[0.0])) #a base rate, b coefficient of drug 
    cov::T2 = () #no covariates required
    timeframe = (general_timeframe = true, inherent_timeframe = 1.0)
    autocorrelation = (false, 0.0)
end

#Outer Constructor to make default for N drugs
function SeizureBasic(N::Int64)
    B = [0.0 for i in 1:N]
    obj = SeizureBasic(θ = ComponentArray((a = 2.0, b = SA[B...])))
    return obj
end

#basic distribution function for record_interval in days starting at n, requires sol from chosen PK model
function distribution(m::SeizureBasic, sol, n::AbstractFloat; person::Union{Person, Nothing} = nothing, record_interval::AbstractFloat = 1.0, names::NamedTuple, θ::ComponentArray = m.θ)
    if any(x -> !isfinite(x), (sol(n+record_interval, idxs = names.S)-sol(n,idxs = names.S)))
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
    timeframe = (general_timeframe = false, inherent_timeframe = 1.0)
    autocorrelation = (true, 1.0)
end

#Outer Constructor to make default for N drugs
function SeizureNegativeBinomial(N::Int64)
    obj = SeizureNegativeBinomial(θ = ComponentArray((a = 2.0, o = 0.01, prev = 0.0, b = SA[0 for i in 1:N])))
    return obj
end

#negative binomial distribution function for record_interval starting at n, depends on if seizure on previous interval, requires sol from chosen PK model
function distribution(m::SeizureNegativeBinomial, sol, n::AbstractFloat; person::Person, seizures::AbstractVector = person.seizure_counts, names::NamedTuple, θ::ComponentArray = m.θ)
    if any(x -> !isfinite(x), (sol(n+1, idxs = names.S)-sol(n,idxs = names.S)))
        return nothing
    end
    o = θ.o
    if o ≤ zero(o) || !isfinite(o)
        return nothing
    end
    seizure_prev_day = (0 < sum([seizure.count for seizure in seizures if (n-1 ≤ seizure.time[1] && seizure.time[2] ≤ n)]))
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

function log_Seizure_prob_instance(m::SeizureModelNonrandom, sol; person::Person, seizures::AbstractVector = person.seizure_counts, prev_seizures::AbstractVector = seizures, names::NamedTuple, θ::ComponentArray = m.θ)
    prob = zero(eltype(θ))   
    for seizure in seizures
        if !(m.autocorrelation[1])
            distribute = distribution(m, sol, seizure.time[1], person = person, names=names, θ = θ)
        else
            distribute = distribution(m, sol, seizure.time[1], person = person, seizures = prev_seizures, names=names, θ = θ)
        end
        if isnothing(distribute)
            return -Inf
        end
        if (seizure.count isa Bool) && (eltype(distribute) != Bool)
            prob_false = pdf(distribute, 0)
            if seizure.count 
                prob += log(1-prob_false)
            else
                prob += log(prob_false)
            end
        else
            prob += log(pdf(distribute, seizure.count))
        end
    end
    return prob
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
    if m.timeframe.general_timeframe
        return log_Seizure_prob(m, sol, person, θ=θ, names = names)
    elseif !(m.autocorrelation[1])
        prob = zero(eltype(θ))
        for i in eachindex(person.seizure_counts) 
            @inbounds time = person.seizure_counts[i].time #get time interval out of named tuple
            @inbounds count = person.seizure_counts[i].count #get count out of tuple
            start = time[1]
            interval = time[2] - time[1]
            if interval < 0
                error("Seizure time interval bounds in wrong order")
            end
            multiple = Int(round(interval/m.timeframe.inherent_timeframe)) #how many timeframes of model occur in interval
            if count isa Bool
                #calculate likelihood of no seizures occuring in timeframe multiple
                seizures = [(time = start+(j-1)*m.timeframe.inherent_timeframe, count = zero(eltype(θ))) for j in 1:multiple]
                log_prob_none = log_Seizure_prob_instance(m, sol, person = person, seizures = seizures, names = names, θ = θ)
                if !isfinite(log_prob_none)
                    return -Inf
                end
                if count
                    prob += log(1-exp(log_prob_none))
                else
                    prob += log_prob_none
                end
            else
                count_sum = zero(eltype(θ)) #storing sum over combinations for this count (sum outside log)
                for counts in multiexponents(multiple, count) #sum over possible combinations for this count over multiple timeframes
                    seizures = [(time = start+(j-1)*m.timeframe.inherent_timeframe, count = counts[j]) for j in 1:multiple]
                    log_interval_prob = log_Seizure_prob_instance(m, sol, person = person, seizures = seizures, names = names, θ=θ)
                    if !isfinite(log_interval_prob)
                        return -Inf
                    else
                        count_sum += exp(log_interval_prob)
                    end
                end
                prob += log(count_sum)
            end
        end
        return prob
    else
        prob = zero(eltype(θ))
        if isempty(person.seizure_counts)
            return prob
        end
        if !(person.seizure_counts[1].count isa Bool)
            #First collect all possible combinations leading to this summarised output
            per_timeframe = Tuple(multiexponents(Int(round((entry.time[2]-entry.time[1])/m.timeframe.inherent_timeframe)), entry.count) for entry in person.seizure_counts)
        else
            #Same for boolean input
            per_timeframe = Tuple(entry.count ? filter(any, collect(Iterators.product(fill((true, false),Int(round((entry.time[2]-entry.time[1])/m.timeframe.inherent_timeframe)))...))) : [(fill(false,Int(round((entry.time[2]-entry.time[1])/m.timeframe.inherent_timeframe))),)] for entry in person.seizure_counts)
        end
        start = person.seizure_counts[1].time[1]
        sums = []
        len = 1
        prev_seizures = []
        start_current = start
        for i in eachindex(per_timeframe)
            if !isempty(prev_seizures)
                start_current = start + length(prev_seizures[1])*m.timeframe.inherent_timeframe
            end
            len = len * length(per_timeframe[i])
            sums_prev = sums
            sums = Array{AbstractFloat}(undef, len)
            a = collect(per_timeframe[i])
            if i == 1 
                for j in eachindex(a)
                    current = [(time = start_current+(k-1)*m.timeframe.inherent_timeframe, count = a[j][k]) for k in eachindex(a[j])]
                    sums[j] = log_Seizure_prob_instance(m, sol, person = person, seizures = current, prev_seizures = current, names = names, θ=θ)
                end
                prev_seizures = a
            else
                for prev in eachindex(prev_seizures)
                    previous = [(time = start+(k-1)*m.timeframe.inherent_timeframe, count = prev_seizures[prev][k]) for k in eachindex(prev_seizures[prev])]
                    for j in eachindex(a)
                        current = [(time = start_current+(k-1)*m.timeframe.inherent_timeframe, count = a[j][k]) for k in eachindex(a[j])]
                        sums[(1-prev)*length(a)+j] = sums_prev[prev] + log_Seizure_prob_instance(m, sol, person = person, seizures = current, prev_seizures = vcat(previous,current), names = names, θ=θ)
                    end
                end
                prev_seizures = [(previous...,option...) for previous in prev_seizures for option in a]
            end
        end
        #Finally sum (outside logscale) over all possible paths, then convert back to log
        return log(sum(exp.(sums)))
    end
end

#3) Implement generation of seizures for discrete, nonrandom models

#models need attribute timeframe = (general_timeframe = yes/no, inherent_timeframe = length in days e.g. 1.0)
#models need bool attribute autocorrelation, length if yes, i.e. autocorrelation = (yes/no, timeframe)
#distribution only needs attribute record_interval if have general_timeframe
function generate_seizures!(m::SeizureModelNonrandom, sol, person::Person; timepoints::AbstractVector=0.0:1.0:10.0, just_Bool::Bool = false, generate_in_lumps::Bool = true, names::NamedTuple)
    if m.timeframe.general_timeframe && generate_in_lumps
        if !(just_Bool)
            new_seizures = [(time = (timepoints[i], timepoints[i+1]), count = rand(distribution(m, sol, timepoints[i], record_interval = timepoints[i+1]-timepoints[i], person = person, names=names))) for i in 1:(length(timepoints)-1)]
        else
            new_seizures = [(time = (timepoints[i], timepoints[i+1]), count = (rand(distribution(m, sol, timepoints[i], record_interval = timepoints[i+1]-timepoints[i], person = person, names=names))>0)) for i in 1:(length(timepoints)-1)]
        end
        append!(person.seizure_counts, new_seizures)
        return
    end
    #Check timepoints admissable, can get weird for Floats
    steps = m.timeframe.inherent_timeframe
    admissable = true
    for i in eachindex(timepoints)
        if i == length(timepoints) || !admissable
            break
        end
        interval = timepoints[i+1] - timepoints[i]
        admissable = admissable && isapprox(round(interval / steps) * steps, interval)
    end
    if !(admissable)
        error("Given timepoints are not compatible with inherent timeframe of model")
    end
    #Generate first in inherent timeframe
    endpoint = max(timepoints...)
    if !(m.autocorrelation[1])
        #generate all directly
        new_seizures = [(time = (j, j+steps), count = rand(distribution(m, sol, j, person = person, names=names))) for j in timepoints[1]:steps:(endpoint-steps)]
        #summarise seizures now and append
        if !(just_Bool)
            summarised = [(time = (timepoints[i], timepoints[i+1]), count = sum([seizure.count for seizure in new_seizures if (timepoints[i] ≤ seizure.time[1] && seizure.time[2] ≤ timepoints[i+1])])) for i in 1:(length(timepoints)-1)]
        else
            summarised = [(time = (timepoints[i], timepoints[i+1]), count = (0<sum([seizure.count for seizure in new_seizures if (timepoints[i] ≤ seizure.time[1] && seizure.time[2] ≤ timepoints[i+1])]))) for i in 1:(length(timepoints)-1)]
        end
        append!(person.seizure_counts, summarised)
    else
        #need info about previous seizures for autocorrelation, append to person stepwise, need to summarise after all data generation done
        for j in timepoints[1]:steps:(endpoint-steps)
            append!(person.seizure_counts, [(time = (j, j+steps), count = rand(distribution(m, sol, j, person = person, names=names)))])
        end
    end
end

function summarise_seizures!(m::SeizureModelNonrandom, person::Person; timepoints::AbstractVector=0.0:1.0:10.0, just_Bool::Bool = false)
    if !m.autocorrelation[1]
        return
    end
    if !(just_Bool)
        summarised = [(time = (timepoints[i], timepoints[i+1]), count = sum([seizure.count for seizure in person.seizure_counts if (timepoints[i] ≤ seizure.time[1] && seizure.time[2] ≤ timepoints[i+1])])) for i in 1:(length(timepoints)-1)]
    else
        summarised = [(time = (timepoints[i], timepoints[i+1]), count = (0<sum([seizure.count for seizure in person.seizure_counts if (timepoints[i] ≤ seizure.time[1] && seizure.time[2] ≤ timepoints[i+1])]))) for i in 1:(length(timepoints)-1)]
    end
    empty!(person.seizure_counts)
    append!(person.seizure_counts, summarised)
end

#4) Functions for visualisation

#Plots fit for person i up to time given solutions from PK model (if they should be plotted)
#estimate and true distributions plotted in same timeframes as individuals data
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
    for i in individuals
        #Get timepoints and corresponding indices for plotting from seizure data
        indices = [index for index in eachindex(data[i].seizure_counts) if data[i].seizure_counts[index].time[1] >= time[1] && data[i].seizure_counts[index].time[2] <= time[2]]
        intervals = [data[i].seizure_counts[index].time for index in indices]
        pl2 = plot(xlabel = "day", ylabel = "Seizure Probability", title = "Seizure probabilities for person $(i) for intervals from $(time[1]) to $(time[2])")
        if !isnothing(sols)
            samples_true = [draw_data_samples(mod, sols[i], person=data[i], interval=interval,names=names) for interval in intervals]
            if any(isnothing.(samples_true))
                @warn "Distribution for true parameters is not well-defined"
                sols = nothing
            end
        end
        if !isnothing(estimate_param)
            samples_estimate = [draw_data_samples(mod, sols2[i], person=data[i], interval=interval, names=names, θ = estimate_param) for interval in intervals]
            if any(isnothing.(samples_estimate))
                @warn "Distribution for estimate parameters is not well-defined"
                estimate_param = nothing
            end
        end
        #Plot distributions, first day separate so label only once, only on separate sides if not both plotted
        if !isnothing(estimate_param) && !isnothing(sols)
            violin!(["$(intervals[1])"], Float64.(samples_true[1]), side = :left, label = "true", colour = :dodgerblue)
            violin!(["$(intervals[1])"], Float64.(samples_estimate[1]), side = :right, label = "estimate", colour = :firebrick2)
            for j in eachindex(intervals)
                if j>1
                    violin!(["$(intervals[j])"], Float64.(samples_true[j]), side = :left, label = "", colour = :dodgerblue)
                    violin!(["$(intervals[j])"], Float64.(samples_estimate[j]), side = :right, label = "", colour = :firebrick2)
                end
            end
        elseif !isnothing(estimate_param)
            violin!(["$(intervals[1])"], Float64.(samples_estimate[1]), label = "estimate", colour = :firebrick2)
            for j in eachindex(intervals)
                if j>1
                    violin!(["$(intervals[j])"], Float64.(samples_estimate[j]), label = "", colour = :firebrick2)
                end
            end
        elseif !isnothing(sols)
            violin!(["$(intervals[1])"], Float64.(samples_true[1]), label = "true", colour = :dodgerblue)
            for j in eachindex(intervals)
                if j>1
                    violin!(["$(intervals[j])"], Float64.(samples_true[j]), label = "", colour = :dodgerblue)
                end
            end
        end
        #Add where data is
        boxplot!(["$(intervals[1])"], [Float64(data[i].seizure_counts[indices[1]].count)], label = "Data values", colour = :grey, linewidth = 3)
        for j in eachindex(intervals)
            if j>1
                boxplot!(["$(intervals[j])"], [Float64(data[i].seizure_counts[indices[j]].count)], label = "", colour = :grey, linewidth = 3)
            end
        end
        #add to output
        append!(output, [pl2])
        if display_plot
            display(pl2)
        end
    end
    return output
end

function draw_data_samples(mod::SeizureModel, sol; person::Person, interval::Tuple, names::NamedTuple, θ::ComponentArray = mod.θ, sample_nr::Int = 1000)
    if mod.timeframe.general_timeframe
        distribute = distribution(mod, sol, interval[1], person=person, record_interval=(interval[2]-interval[1]),names=names, θ=θ)
        if isnothing(distribute)
            return nothing
        end
        return rand(distribute,sample_nr)
    elseif !mod.autocorrelation[1]
        steps = mod.timeframe.inherent_timeframe
        distributions = [distribution(mod, sol, j, person = person, names=names, θ=θ) for j in interval[1]:steps:(interval[2]-steps)]
        if any(isnothing.(distributions))
            return nothing
        end
        #Simulate data for each day, then sum vectors over days
        simulate = [rand(distribute, sample_nr) for distribute in distributions]
        return sum(simulate)
    else
        steps = mod.timeframe.inherent_timeframe
        #retrieve relevant previous seizure data (within autocorrelation time)
        prev_seizures = [seizure for seizure in person.seizure_counts if (interval[1]-mod.autocorrelation[2]) < seizure.time[2] <= interval[1]]
        if isempty(prev_seizures)
            #draw and append with no previous, simulate one sample path for each sample_nr
            seizures = [deepcopy(prev_seizures) for i in 1:sample_nr]
            for j in interval[1]:steps:(interval[2]-steps)
                distributions = [distribution(mod, sol, j, person=person, seizures = path, names = names, θ=θ) for path in seizures]
                if any(isnothing.(distributions))
                    return nothing
                end
                samples = [(time = (j, (j+steps)), count = rand(distribute)) for distribute in distributions]
                for i in eachindex(seizures)
                    push!(seizures[i], samples[i])
                end
            end
            #just want to sum over count in each path
            sums = [sum([seizure.count for seizure in seizures_path]) for seizures_path in seizures]
            return sums
        else
            if !(prev_seizures[1].count isa Bool)
                #First collect all possible combinations leading to this summarised output
                per_timeframe = (multiexponents(Int(round((entry.time[2]-entry.time[1])/steps)), entry.count) for entry in prev_seizures)
            else
                #Same for boolean input
                per_timeframe = (entry.count ? filter(any, collect(Iterators.product(fill((true, false),Int(round((entry.time[2]-entry.time[1])/steps)))...))) : (fill(false,Int(round((entry.time[2]-entry.time[1])/steps))),) for entry in prev_seizures)
            end
            #sum over products, calculate how many samples per combi, then generate
            start = prev_seizures[1].time[1]
            per_combi = Int(ceil(sample_nr/length(Iterators.product(per_timeframe...))))
            simulate = Vector{Int}()
            for combi in Iterators.product(per_timeframe...)
                combi = collect(Iterators.flatten(combi))
                seizures = [(time = (start+(j-1)*mod.timeframe.inherent_timeframe, start+j*mod.timeframe.inherent_timeframe), count = combi[j]) for j in eachindex(combi)]
                #now simulate draw for each possible combination in prev_seizures, per_combi samples each
                seizures = [deepcopy(seizures) for i in 1:per_combi]
                for j in interval[1]:steps:(interval[2]-steps)
                    distributions = [distribution(mod, sol, j, person=person, seizures = path, names = names, θ=θ) for path in seizures]
                    if any(isnothing.(distributions))
                        return nothing
                    end
                    samples = [(time = (j, j+steps), count = rand(distribute)) for distribute in distributions]
                    #append samples in format of seizure counts 
                    push!.(seizures,samples)
                end
                #sum just over count and only within our interval
                seizures = [sum([seizure.count for seizure in seizures_path if (interval[1] <= seizure.time[1] && interval[2] >= seizure.time[2])]) for seizures_path in seizures]
                append!(simulate, seizures)
            end
            return simulate
        end
    end
end