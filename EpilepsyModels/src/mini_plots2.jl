using Pkg

#Pkg.instantiate()
println("Starting")
#Pkg.develop(path = ".//EpilepsyModels")
include("EpilepsyModels.jl")

using .EpilepsyModels
using ComponentArrays
using OptimizationOptimJL
using OptimizationBBO
using LineSearches
using DifferentialEquations
using Plots
using StatsPlots
using StaticArrays
using Random
using Distributions
using BenchmarkTools
using ModelingToolkit
using MCMCChains
using AdvancedHMC

#This will redirect output to txt file, not including error messages/warnings
#path = "."
#path = "/home/s6newell_hpc"
#open(joinpath(path,"output.txt"), "w") do io
#open(joinpath(path,"errors.txt"), "w") do io_err
#redirect_stdout(io) do
#redirect_stderr(io_err) do

println("Included")

#try

#set seed
Random.seed!(42)

#Specify parameters for models
#PK Models
Input_θ_PKBasic = ComponentArray((k_el = 2.0, k_abs = 5.0, σ=0.2))
Input_θ_PKLEV = ComponentArray((k_abs = (24*3.5), c1 = (24*4.0), c2 = 0.25, c3 = 0.122, v1 = 29.7, v2 = 2.85, σ=0.2))
Input_θ_PKLEVNoAbsorption = ComponentArray((c1 = (24*4.0), c2 = 0.25, c3 = 0.122, v1 = 29.7, v2 = 2.85, σ=0.2))
Input_θ_PKCBZ = ComponentArray((k_abs = (24*0.45), c1 = (24*1.96), c2 = 1.73, c3 = 24*1.36, v1 = 164.0/75.0, σ=0.2))
Input_θ_PKVPA = ComponentArray((k_abs = (24*1.86), c1 = (24*0.577), c2 = 0.535, c3 = 0.875, v1 = 0.28, σ=0.2))
Input_θ_PKLTG = ComponentArray((k_abs = (24*1.96), c1 = (24*2.4), c2 = 0.938, c3 = 110*0.00328, c4 = 0.34, v1 = 2.14, σ=0.2))
Input_θ_PKBigFour = ComponentArray((k_abs_LTG = (24*1.96), c1_LTG = (24*2.4), c2_LTG = 0.938, c3_LTG = 110*0.00328, c4_LTG = 0.34, c_Inh_LTG = (1-0.579), c_Ind_LTG = (1+0.546), v1_LTG = 2.14, σ_LTG=0.2,
            k_abs_VPA = (24*1.86), c1_VPA = (24*0.577), c2_VPA = 0.535, c3_VPA = 0.875, c_Ind_VPA = 1.22, v1_VPA = 0.28, σ_VPA=0.2,
            k_abs_CBZ = (24*0.45), c1_CBZ = (24*1.96), c2_CBZ = 1.73, c3_CBZ = 24*1.36, v1_CBZ = 164.0/75.0, σ_CBZ=0.2,
            k_abs_LEV = (24*3.5), c1_LEV = (24*4.0), c2_LEV = 0.25, c3_LEV = 0.122, c_Inh_LEV = 0.812, c_Ind_LEV = 1.09, v1_LEV = 29.7, v2_LEV = 2.85, σ_LEV=0.2))
#Seizure Models
base_rate = 4.0
Input_θ_SeizureBasic_one = ComponentArray((a = base_rate, b = SA[0.2]))
Input_θ_SeizureBasic_four = ComponentArray((a = base_rate, b = SA[base_rate/7, base_rate/25, base_rate/20, base_rate/120]))
#Input_θ_SeizureNegativeBinomial = ComponentArray((a = -1.923, o = 1.128, prev = 0.731, b = SA[0.2]))
Input_θ_SeizureNegativeBinomial = ComponentArray((a = log(4.0), o = 1.128, prev = 0.731, b = SA[0.2]))
Input_θ_SeizureVPA = ComponentArray((a = 6.1, a1 = 1.0, a2 = 1.8, b1 = 13.3, b2 = 2.4))
Input_θ_SeizureSANAD_one = ComponentArray((a1 = log(1.09), a2 = log(0.87), a3 = log(1.15), b = SA[7/30])) 
Input_θ_SeizureSANAD_four = ComponentArray((a1 = log(1.09), a2 = log(0.87), a3 = log(1.15), b = SA[6.75/9, 7.5/29, 7.25/8, 7.25/75]))

Maxiters_optimiser = 200
Samples_per_chain = 2000
Adaptation_steps = 1000
Max_Time = 20*60.0 #14*60.0*60.0 
#Max_Time = 23.5*60*60 #4.0*60*60 + 30.0*60 #maximal optimisertime in seconds
Population_size = 20 #5 #20 #10 #20
wo_treatment = 0.0 #10.0
Obs_Duration = wo_treatment + 10.0 #40.0
PK_timepoints = (wo_treatment+0.5):2.25:(Obs_Duration-0.5) #[0.2, 0.35, 1.05, 2.3, 3.7, 4.6, 5.75, 6.01, 7.5, 8.9]
#TODO Change this back later
Seizure_timepoints = 0.0:5.0:Obs_Duration
max_events = nothing
no_counts_seizure = false
#logscale = ("σ",)
#logscale = ("σ","v1")
#logscale = ("σ", "k_abs", "c1", "v1", "a")
logscale = ("σ", "k_abs", "c1", "v1")
#logscale = ("σ_LEV", "k_abs_LEV", "c1_LEV", "v1_LEV", "σ_LTG", "k_abs_LTG", "c1_LTG", "v1_LTG","σ_CBZ", "k_abs_CBZ", "c1_CBZ", "v1_CBZ", "σ_VPA", "k_abs_VPA", "c1_VPA", "v1_VPA", "a")
#logscale = ("σ", "k_abs", "c1", "c3", "v1", "a") 
#logscale = ("σ", "c1", "v1", "a")
println("logscale = ", logscale)
solver_optim = LBFGS(linesearch = LineSearches.BackTracking())
#solver_optim = BBO_adaptive_de_rand_1_bin_radiuslimited()
sampler = NUTS(0.8) #nothing
sampling_options = (verbose = false, progress = false) #limits output
ODE_options = (AutoTsit5(Rosenbrock23()),)
#ODE_options = (Rosenbrock23(),)

#Multistart settings (LHS) for robust optimisation from weak/default initial guesses.
#All bounds are in transformed space (i.e. log-scale for logscale parameters).
max_threads_simul = Threads.nthreads()
Multistart_nstarts = 1
prefilter = 10
Multistart_seed = 42
Multistart_include_initial = true
custom_starts = nothing
bound_abs = nothing #100.0
optim_lower_bounds = nothing
optim_upper_bounds = nothing
Variance_bound = nothing #log(1.0) #upper bounds will be reset accordingly after Input_θ is created below
Multistart_bounds = nothing #25.0
fail_hard = false

run_hessian = false
sandwich = true
finite_diff_hessian = false
drug_appropriate_dosing = true
hierarchical_optimisation = false
sampling = false
plotting = false
optimisation_trace = true
show_trace = true

#pk_model = PKBasic(θ=Input_θ_PKBasic)
#pk_model = PKLEV(θ=Input_θ_PKLEV)
#pk_model = PKLEVNoAbsorption(θ=Input_θ_PKLEVNoAbsorption)
#pk_model = PKCBZ(θ=Input_θ_PKCBZ)
#pk_model = PKVPA(θ=Input_θ_PKVPA)
#pk_model = PKLTG(θ=Input_θ_PKLTG)
pk_model = PKBigFour(θ = Input_θ_PKBigFour)
#Set b in seizure_basic according to pk model (different daily exposures), for VPA 0.2 is too high

if (typeof(pk_model).name.wrapper in [PKVPA])
    Input_θ_SeizureBasic_one.b = SA[0.05]
end
if (typeof(pk_model).name.wrapper in [PKBigFour])
    seizure_model = SeizureBasic(θ = Input_θ_SeizureBasic_four)
else
    if drug_appropriate_dosing
        if (typeof(pk_model).name.wrapper in [PKLEV, PKLEVNoAbsorption])
            Input_θ_SeizureBasic_one.b = SA[Input_θ_SeizureBasic_four.b[2]]
        elseif (typeof(pk_model).name.wrapper in [PKLTG])
            Input_θ_SeizureBasic_one.b = SA[Input_θ_SeizureBasic_four.b[1]]
        elseif (typeof(pk_model).name.wrapper in [PKCBZ])
            Input_θ_SeizureBasic_one.b = SA[Input_θ_SeizureBasic_four.b[3]]
        elseif (typeof(pk_model).name.wrapper in [PKVPA])
            Input_θ_SeizureBasic_one.b = SA[Input_θ_SeizureBasic_four.b[4]]
        end
    end
    seizure_model = SeizureBasic(θ = Input_θ_SeizureBasic_one)
end

#seizure_model = SeizureMult(pk_model, base_rate = base_rate, default_treat_eff = 0.2)
#seizure_model = SeizureVPA(θ = Input_θ_SeizureVPA)
#seizure_model = SeizureNegativeBinomial(θ = Input_θ_SeizureNegativeBinomial)
#SeizureSANAD set b appropriately
#=
if (typeof(pk_model).name.wrapper in [PKBigFour])
    seizure_model = SeizureSANAD(θ = Input_θ_SeizureSANAD_four, baseline = base_rate)
else
    if drug_appropriate_dosing
        if (typeof(pk_model).name.wrapper in [PKLEV, PKLEVNoAbsorption])
            Input_θ_SeizureSANAD_one.b = SA[Input_θ_SeizureSANAD_four.b[2]]
        elseif (typeof(pk_model).name.wrapper in [PKLTG])
            Input_θ_SeizureSANAD_one.b = SA[Input_θ_SeizureSANAD_four.b[1]]
        elseif (typeof(pk_model).name.wrapper in [PKCBZ])
            Input_θ_SeizureSANAD_one.b = SA[Input_θ_SeizureSANAD_four.b[3]]
        elseif (typeof(pk_model).name.wrapper in [PKVPA])
            Input_θ_SeizureSANAD_one.b = SA[Input_θ_SeizureSANAD_four.b[4]]
        end
    end
    seizure_model = SeizureSANAD(θ = Input_θ_SeizureSANAD_one, baseline = base_rate)
end
=#

person_gen = BigFourPersonGenerator()
#dose_gen = BasicDoses(default_dose=500.0, times_per_day=2)
#dose_gen = PolyDoses(pk_model, default_dose=500.0)
#Create appropriate dose generator based on which pk_model was chosen
dose_gen = PolyDosesRandom(pk_model, drug_appropriate_dosing)
#dose_gen = BigFourDoses()
if seizure_model isa SeizureVPA
    dose_distr = (d_VPA = (min = 150.0, avg_num = 8.0, max_num = 14), d_CBZ = (min = 200.0, avg_num = 3.0, max_num = 8))
    distr_first = (d_VPA = 1.0, d_CBZ = 0.0)
    distr_second = (d_VPA = 0.0, d_CBZ = 1.0)
    dose_gen = PolyDosesRandom(dose_distr, distr_first, distr_second; prob_second=0.5, times_per_day_first=2, times_per_day_second=2, assign_not_supported = true)
    #dose_gen = BigFourDoses(order_male = ((:d_VPA,:d_CBZ), (:d_VPA,:d_CBZ)), order_female = ((:d_VPA,:d_CBZ), (:d_VPA,:d_CBZ)), prob_second = 1.0)
end
Input_θ = ComponentArray(PK = pk_model.θ, Seizure = seizure_model.θ)
mod = FullModel(pk_model, seizure_model, person_gen, dose_gen)
data = generate_data(mod, Population_size, Obs_Duration, timepoints_PK = PK_timepoints, timepoints_seizure = Seizure_timepoints, max_events = max_events, wo_treatment = wo_treatment, max_threads = max_threads_simul, just_Bool = no_counts_seizure, ODE_options = ODE_options)
#modifications = ((2, Normal(0, Input_θ[2]/4)),(label2index(Input_θ,"Seizure.a")[1], Normal(0,Input_θ.Seizure.a/6)))
#data = generate_data_modified(mod, Population_size, Obs_Duration, update_reg = 5.0, modifications = modifications, timepoints_PK = PK_timepoints, timepoints_seizure = Seizure_timepoints, max_threads = max_threads_simul, just_Bool = no_counts_seizure, wo_treatment = wo_treatment, ODE_options = ODE_options)
#data = generate_data_updating(mod, Population_size, Obs_Duration, update_reg = 5.0, timepoints_PK = PK_timepoints, timepoints_seizure = Seizure_timepoints, max_threads = max_threads_simul, just_Bool = no_counts_seizure, wo_treatment = wo_treatment, ODE_options = ODE_options)
println("Generated")

thickness = 2
endpoint = 21.0
empty!(data[1].dosing)
next_doses = [(t = i+1 + j/2, dose = 500.0, state = :d_CBZ) for i in -1:20 for j in 0:(2-1)]
append!(data[1].dosing, next_doses)
#=
next_doses = [(t = i+1 + j/2, dose = 750.0, state = :d_LEV) for i in 7:13 for j in 0:(2-1)]
append!(data[1].dosing, next_doses)
next_doses = [(t = i+1 + j/2, dose = 200.0, state = :d_LTG) for i in 14:20 for j in 0:(2-1)]
append!(data[1].dosing, next_doses)
=#
sol = EpilepsyModels.solve_PK(pk_model, pk_model.θ, data[1], endpoint=endpoint)
pl = plot(xlabel="Time", ylabel="Concentration", tspan = (0, endpoint), ticks = false, thickness_scaling=thickness)
#plot!(sol, idxs = :s_LEV, label="", tspan = (0,endpoint), linewidth = 2, colour = :green)
plot!(sol, idxs = :s_CBZ, label="", tspan = (0,endpoint), linewidth = 2, colour = :blue, ylims = (0,20))
#plot!(sol, idxs = :s_LTG, label="", tspan = (0,endpoint), linewidth = 2, colour = :purple)
display(pl)
#=
x_values = [measurement.timepoint for measurement in data[1].measurements if (measurement.state[2] == pk_model.keys.s[1])]
y_values = [measurement.measurement for measurement in data[1].measurements if (measurement.state[2] == pk_model.keys.s[1])]
plot!(x_values, y_values, seriestype = :scatter, mc = :purple, markersize = 5, label = "", tspan = (0,endpoint))
display(pl)
=#

#Bell curves for Modifiers
function normal_pdf(μ, σ, x)
    return 1/sqrt(2*pi*(σ^2))*exp(-(x-μ)^2/(2*σ^2))
end

thickness = 2
normals = (([4.0, 1.0],), ([4.0, 2.0],), ([4.0, 4.0],),
            ([4.0, 1.0],[10.0, 1.0]), ([4.0, 2.0],[10.0, 3.0]), ([4.0, 4.0],[10.0, 5.0]),
            ([4.0, 1.0],[10.0, 1.0], [15.0, 2.0]), ([4.0, 2.0],[10.0, 3.0], [15.0, 5.0]), ([4.0, 4.0],[10.0, 5.0], [15.0, 8.0]))

colours = (:blue, :red, :green)
endpoint = 30.0
output = Plots.Plot[]
for j in eachindex(normals)
    pl = plot(xlabel="Value", ylabel="Density", xlims = (0, endpoint), ylims = (0, 0.4), ticks = true, grid = false, thickness_scaling=thickness,
                guidefontsize = 8, legendfontsize = 6, right_margin = 3Plots.mm, left_margin = 0Plots.mm)
    for i in eachindex(normals[j])
        current_pdf(x) = normal_pdf(normals[j][i][1], normals[j][i][2],x)
        plot!(current_pdf, fillrange = 0, fc=colours[i], label = "", alpha = 0.5)
        plot!(current_pdf, label="Parameter $(i)", linewidth = 1.5, colour = colours[i])
        vline!([normals[j][i][1]], linecolor = colours[i], label = "", linewidth=2)
    end
    display(pl)
    push!(output, pl)
end
names = (:small, :medium, :wide)
for i in eachindex(output)
    section = Int(ceil(i/3))
    println("Section:", section)
    width = i - ((section-1)*3)
    println("Width:", width)
    println()
    savefig(output[i], "./Modifiers/$(section)_$(names[width]).png")
end