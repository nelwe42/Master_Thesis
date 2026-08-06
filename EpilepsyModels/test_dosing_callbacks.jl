#Run with: julia --project=. test_dosing_callbacks.jl
#Checks that the single all-doses callback injects every dose, and that a long observation
#duration (which used to overflow the compiler's stack with one callback per dose) still works.
using EpilepsyModels
const EM = EpilepsyModels

#1) every scheduled dose is injected into the right depot, including one exactly at tspan[1]
pk = EM.PKVPA()
dosing = [(t = i + j/2, dose = 300.0, state = :d_VPA) for i in 0.0:4.0 for j in 0:1]
sol = EM.solve_ODE(pk, dosing = dosing, covariates = (gender = 1.0, weight = 70.0), endpoint = 5.0)
@assert EM.SciMLBase.successful_retcode(sol)
for dose in dosing
    before = dose.t > 0 ? sol(dose.t - 1e-8, idxs = :d_VPA) : 0.0
    @assert isapprox(sol(dose.t + 1e-8, idxs = :d_VPA) - before, 300.0, rtol = 1e-5) "no dose at t=$(dose.t)"
end

#2) doses of several drugs at the same timepoint all arrive, doses of unsupported drugs are ignored
pk4 = EM.PKBigFour()
cov4 = (weight = 70.0, height = 175.0, kidney_disease = 0.0, CLCr = 100.0, smoking = 0.0, prev_CBZ = 0.0, gender = 1.0)
same_time = [(t = 1.0, dose = 300.0, state = :d_VPA), (t = 1.0, dose = 200.0, state = :d_CBZ),
             (t = 1.0, dose = 999.0, state = :d_not_in_model)]
sol4 = EM.solve_ODE(pk4, dosing = same_time, covariates = cov4, endpoint = 2.0)
@assert isapprox(sol4(1.0 + 1e-8, idxs = :d_VPA), 300.0, rtol = 1e-5)
@assert isapprox(sol4(1.0 + 1e-8, idxs = :d_CBZ), 200.0, rtol = 1e-5)

#3) long observation duration with dose, daily dose and autoinduction callbacks
long_dosing = [(t = i + j/2, dose = 300.0, state = d) for i in 0.0:99.0 for d in (:d_VPA, :d_CBZ) for j in 0:1]
prob = EM.create_problem(pk4, dosing = long_dosing, covariates = cov4, endpoint = 100.0)
@assert length(prob.kwargs[:callback].discrete_callbacks) == 4 #doses + 2 daily doses + 1 autoinduction
@assert EM.SciMLBase.successful_retcode(EM.solve_ODE(pk4, dosing = long_dosing, covariates = cov4, endpoint = 100.0))

println("dosing callback tests passed")
