using ModelingToolkit
using DifferentialEquations

#Theta_PK = [k_el, k_abs], Dose=dose (once daily), endpoint = until when ODE problem
function ODE_system(Theta_PK::Vector{<:Real}, Dose; endpoint = 10)
    @mtkmodel Internal begin
        @parameters begin
            k_el
            k_abs
            dose
        end
        @variables begin
            d(t) = dose #drug dose left
            i(t) = 0 #internal state
        end
        @equations begin
            D(d) ~ -k_abs*d
            D(i) ~ k_abs*d - k_el*i
        end
        @discrete_events begin
            (floor(t) == ceil(t)) => [d ~ Pre(d) + dose] #each day new dose taken
            #should maybe insert after certain time remnant can no longer be absorped
        end
    end
    @mtkcompile internal_model = Internal(; k_el = Theta_PK[1], k_abs = Theta_PK[2], dose = Dose)
    problem = ODEProblem(internal_model, [], (0,endpoint)) #system, list of reset parameter defaults, timeframe
    return problem
end

print("Done")