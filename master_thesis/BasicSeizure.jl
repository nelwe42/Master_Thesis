using ModelingToolkit
using Distributions
using Random

#set seed
Random.seed!(42)

#Theta_Seizure = [a_0, list of b_d], i(t) = vector of i_d(t) solutions of ODE problem internal, n = day
function intensity(Theta_Seizure, i, n)
    intense = Theta_Seizure[1]
    for d in 1:length(i)
        integral = 0 #solve integral over [n,n+1) of i_d(t) here
        intense = intense + Theta_Seizure[d+1] * integral
    end
    return intense
end

#k_n number of seizures on day n
function Seizure_prob_day(Theta_Seizure, i, n, k_n)
    lambda = intensity(Theta_Seizure, i, n)
    return (lambda^k_n/factorial(k_n))*exp(-lambda)
end

#N vector of days, k vector of number of seizures on days
function Seizure_prob(Theta_Seizure, i, N, k)
    prob = 1
    for n in N #is this allowed syntax in Julia?
        k_n = k[n]
        prob = prob * Seizure_prob_day(Theta_Seizure, i, n, k_n)
    end
    return prob
end

print("Done")