# julia 1.2
push!(LOAD_PATH,pwd())

using PyPlot
# import src
include("src.jl")


# init a model
MC = (
    L = exp.([-0.4, 0.4]), # labor endow states
    P = [0.8 0.2 ; 0.2 0.8],  #transition matrix
    π = [0.5,0.5],  # stationary distribution
)
M = src.ModelBasic( N = 1.0,
BOTTOM_VAL = -6.66E36, AGG_SEED = 20200417,
AGG_SIMU_SIZE = 10000, AGG_DROP_FIRST_SIZE = 1000,
Na = 200, alim = (0.0,4.5),
lgrid = MC.L, P = MC.P, π = MC.π,
θ = 0.36, δ = 0.08, β = 0.96, μ = 2.0 )

# search ss
@time src.solve_ss_gs!(M, 1.0 , max_iter_ss = 500, max_iter_hh = 500,
rtol_ss = 1E-3, rtol_hh = 1E-3, step_size_ss = 0.7, step_size_hh = 0.7,
print_err_ss = true, print_err_hh = false, if_golden = true )


# visualization
plot(M.agrid, M.HHPolicy.aOpt[:,1], linewidth = 0.75)
plot(M.agrid, M.HHPolicy.aOpt[:,2], "--", linewidth = 0.75)
plot(M.agrid, M.agrid, "-.", linewidth = 0.75)
legend(["l - low","l - high", L"$\pi/4$"])






