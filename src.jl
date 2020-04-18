__precompile__()
"""
    src.jl

1. julia <= 1.2
2. StatsBase <= 0.32.0
"""
module src
    using StatsBase, Random
    export ModelBasic, EmptyHouseholdPolicy, HouseholdPolicy
    





# ======================= TYPES
    abstract type AbstractDGEModel <: Any end
    abstract type AbstractPolicyFunctions <: Any end

    # ---------- MODEL
    mutable struct ModelBasic <: AbstractDGEModel
        N::Float64  # total population size
        BOTTOM_VAL::Float64 # bottom value of val func
        AGG_SEED::Int # seed used in aggregation
        AGG_SIMU_SIZE::Int # size of simulation in aggregation
        AGG_DROP_FIRST_SIZE::Int # num of obs to drop in aggregation simulation (to drop non-statinary obs)
        # ------ ECON PARS
        θ::Float64 # cpaital income share
        δ::Float64 # depreciation rate
        β::Float64 # util disc factor
        μ::Float64 # CRRA risk aversion
        # ------ HH VAL FUNC ITER PARS
        Na::Int # num of asset grid units
        Nl::Int # num of labor endow grid units
        NaNl::Int # Na * Nl
        alim::Tuple{Float64,Float64} # asset state space
        agrid::Vector{Float64} # asset grid space
        lgrid::Vector{Float64} # labor endow grid space
        # ------ IDIO LAB ENDOW PROC - MARKOV CHAIN
        P::Matrix{Float64} # one-step trnasition matrix
        π::Vector{Float64} # stationary distribution of labor endow
        # ------ HOUSEHOLD POLICY FUNCTIONS
        HHPolicy::AbstractPolicyFunctions  # household policy functions
        # ------ OTHER ECON DATA
        r::Float64 # net interest rate
        w::Float64 # wage rate
        K::Float64 # agg capital
        L::Float64 # labor factior
        C::Float64 # agg consumption
        Y::Float64 # agg output
        # ------------- CONSTRUCTOR
        """
            ModelBasic( ; N::Float64 = 1.0,
                BOTTOM_VAL::Float64 = -6.66E36, AGG_SEED::Float64 = time(),
                AGG_SIMU_SIZE::Int = 10000, AGG_DROP_FIRST_SIZE::Int = 1000,
                Na::Int = 200, alim::Tuple{Float64, Float64} = (0.0,4.5),
                lgrid::Vector{Float64} = [0.6703,1.4918], P::Matrix{Float64} = [0.8 0.2;0.2 0.8], π::Vector{Float64} = [0.5,0.5],
                HHPolicy::AbstractPolicyFunctions = EmptyHouseholdPolicy(),
                θ::Float64 = 0.36, δ::Float64 = 0.08, β::Float64 = 0.96, μ::Float64 = 2.0 )
        """
        function ModelBasic( ; N::Float64 = 1.0,
            BOTTOM_VAL::Float64 = -6.66E36, AGG_SEED::Int = floor(Int,1E6 * rand()),
            AGG_SIMU_SIZE::Int = 10000, AGG_DROP_FIRST_SIZE::Int = 1000,
            Na::Int = 200, alim::Tuple{Float64, Float64} = (0.0,4.5),
            lgrid::Vector{Float64} = [0.6703,1.4918], P::Matrix{Float64} = [0.8 0.2;0.2 0.8], π::Vector{Float64} = [0.5,0.5],
            HHPolicy::AbstractPolicyFunctions = EmptyHouseholdPolicy(),
            θ::Float64 = 0.36, δ::Float64 = 0.08, β::Float64 = 0.96, μ::Float64 = 2.0 )
            # ---------- 
            local agrid = Vector{Float64}(LinRange(alim[1],alim[2], Na))

            new(N,BOTTOM_VAL,AGG_SEED,AGG_SIMU_SIZE,AGG_DROP_FIRST_SIZE, 
                θ,δ,β,μ,
                Na, length(lgrid), Na * length(lgrid), alim, agrid, lgrid,
                P,π,
                HHPolicy,
                0.1, 1.0, 1.0, 1.0, 1.0, 1.0 ) 
        end # ModelBasic()
    end


    # ---------- HOUSEHOLD POLICY (CLOSURE)
    """
        EmptyHouseholdPolicy <: AbstractPolicyFunctions

    an empty type of household policies;
    used to initialize models.
    """
    struct EmptyHouseholdPolicy <: AbstractPolicyFunctions
    end # EmptyHouseholdPolicy
    # --------------
    """
        HouseholdPolicy <: AbstractPolicyFunctions

    household policy functions type
    """
    struct HouseholdPolicy <: AbstractPolicyFunctions
        Na::Int  # num of asset grid units
        Nl::Int  # num of labor endow grid units
        NaNl::Int  # Na * Nl
        agrid::Vector{Float64} # asset grid
        lgrid::Vector{Float64} # labor endow grid
        aOpt::Matrix{Float64} # decision matrix of asset, Na * Nl
        cOpt::Matrix{Float64} # decision matrix of consumption, Na * Nl
        vMat1::Matrix{Float64} # val func iter result - old guess
        vMat2::Matrix{Float64} # val func iter result - updated
        # --------- CONSTRUCTOR
        """
            HouseholdPolicy(agrid::Vector{Float64}, lgrid::Vector{Float64}, aOpt::Matrix{Float64}, cOpt::Matrix{Float64}, vMat1::Matrix{Float64}, vMat2::Matrix{Float64})
        """
        function HouseholdPolicy(agrid::Vector{Float64}, lgrid::Vector{Float64}, aOpt::Matrix{Float64}, cOpt::Matrix{Float64}, vMat1::Matrix{Float64}, vMat2::Matrix{Float64})
            local Na = length(agrid)
            local Nl = length(lgrid)
            new(Na,Nl,Na*Nl, agrid,lgrid, aOpt,cOpt, vMat1,vMat2)
        end # HouseholdPolicy()
    end # HouseholdPolicy
    # ------ APPENDED METHOD
    """
        eval_hhpolicy_interp(HHP::HouseholdPolicy, val_athis::Float64, idx_lthis::Int)

    evaluates `a(t+1)` at given `val_athis` (asset value of `a(t)`) and `idx_lthis` (index of labor endow state);
    returns a `Tuple{Float64,Float64}` consisting of
    `a(t+1)` and `c(t)` (in order);
    using linear interpolation.
    """
    function eval_hhpolicy_interp(HHP::HouseholdPolicy, val_athis::Float64, idx_lthis::Int)
        local anext = interp1(HHP.agrid, HHP.aOpt[:,idx_lthis], val_athis)
        local cthis = interp1(HHP.agrid, HHP.cOpt[:,idx_lthis], val_athis)
        return (anext,cthis)::Tuple{Float64, Float64}
    end # eval_hhpolicy_interp()




# ======================= NUMERICAL METHODS

    # ------- LINEAR INTERPOLATION (REFACTOR MATLAB interp1)
    function interp1(X::Vector{Float64}, Y::Vector{Float64}, xeval::Float64)
        # require X ascd sorted, at least has 2 ele
        local idx = -1::Int
        for j in 1:(length(X)-1)
            if X[j] <= xeval <= X[j+1]
                idx = j
                break
            end # if
        end # for j
        local res = Y[idx] + (xeval - X[idx])/(X[idx+1] - X[idx] + eps()) * (Y[idx+1] - Y[idx])
        return res::Float64
    end # interp1

    # ------- THREE-POINT GOLDEN-SECTION (HEER & MAUSSNER, 2008)
    function golden_max_3points(f::Function, LLB::Float64, LB::Float64, RB::Float64 ; TOL::Float64 = 1E-6)
        local r1 = 0.61803399
        local r2 = 1 - r1
        local x0 = LLB
        local x3 = RB

        if abs(RB-LB) <= abs(LB-LLB)
            x1 = LB; x2 = LB + r2 * (RB-LB)
        else
            x2 = LB; x1 = LB - r2 * (LB-LLB)
        end # if

        f1 = - f(x1)
        f2 = - f(x2)

        # Searching
        local counter = 1;
        while (  abs(x3-x0) > (TOL * (abs(x1)+abs(x2)))  ) | (counter < 200)
            if f2 < f1
                x0=x1; # Update the very lower bound
                x1=x2;
                x2=r1*x1+r2*x3; # new left golden position
                f1=f2;
                f2= - f(x2);
            else
                x3=x2;
                x2=x1;
                x1=r1*x2+r2*x0;
                f2=f1;
                f1= - f(x1);
            end
            counter = counter +1;
        end
        
        # Post-convergence
        if f1<=f2
            return x1::Float64
        else
            return x2::Float64
        end

        return NaN::Float64
    end # golden_max_3points


# ======================= MODEL-SPECIFIC METHODS

    # ------- UTIL
    ufunc(c::Float64, μ::Float64) = begin
        return ( (c + 1E-8) ^ (1 - μ) / (1 - μ) )::Float64
    end # ufunc
    # ------- INTER-PERIOD BUDGET
    get_c(lthis::Float64, athis::Float64, anext::Float64, w::Float64) = begin
        return (w * lthis + athis - anext)::Float64
    end # get_c
    # ------- DISCOUNTED VAL FUNC (t+1) EVAL
    function vfunc(idx_lthis::Int, val_anext::Float64, mat_vnext::Matrix{Float64}, M::ModelBasic)
        res = 0.0;
        for j in 1:M.Nl
            res += M.P[idx_lthis, j] * interp1(M.agrid, mat_vnext[:,j], val_anext)
        end # for j
        return (M.β * res)::Float64
    end # vfunc
    # ------- BELLMAN BEFORE MAXIMIZATION
    function bellman(val_athis::Float64, idx_lthis::Int, val_anext::Float64, mat_vnext::Matrix{Float64}, M::ModelBasic)
        local cthis = get_c(M.lgrid[idx_lthis], val_athis, val_anext, M.w)
        if cthis < 0
            # (vthis, cthis)
            return ( M.BOTTOM_VAL, 0.0 )::Tuple{Float64, Float64}
        else
            local vthis = ufunc(cthis, M.μ) + vfunc(idx_lthis, val_anext, mat_vnext, M)
            return ( vthis, cthis )::Tuple{Float64, Float64}
        end # if
    end # bellman


# ======================= SIMULATION AGGREGATION METHODS

    # ------------- AGG: VIA SIMU HH PATH
    """
        agg_hh_simu(M::ModelBasic, T::Int, seed::Int ; a0::Float64 = 1.0, drop_first::Int = 100)

    aggregation to get agg variables via simulating a household path (long enough);
    returns a tuple `Tuple{Float64, Float64}` consisting of aggregate capital and aggregate consumption.
    """
    function agg_hh_simu(M::ModelBasic, T::Int, seed::Int ; a0::Float64 = 1.0, drop_first::Int = 100)
        # init
        local avgK = a0
        local avgC = 0.0
        local nowK = a0
        local nowC = 0.0
        # sampling from stationary distribution of labor endowment
        # NOTE: we expect there will not be more than 2^8 states, so use `Int8` to
        #       improve gc performance when T is very large 
        #       (e.g. >= 100,000,000, or equivalent run-time >= 1s)
        vidx_L = sample( MersenneTwister(seed), Vector{Int8}(1:M.Nl), Weights(M.π), T )

        # simulation
        for t in 1:T
            nowK, nowC = eval_hhpolicy_interp(M.HHPolicy, nowK, Int(vidx_L[t]) )
            if t > drop_first
                avgK = avgK + nowK
                avgC = avgC + nowC
            end # if t
        end # t

        # transformation & return
        avgK = avgK / (T - drop_first) * M.N
        avgC = avgC / (T - drop_first) * M.N
        
        return (avgK, avgC)::Tuple{Float64, Float64}
    end # agg_hh_simu()










# ======================= SOLVERS: HOUSEHOLD POLICY FUNCTIONS

    # ----------- HOUSEHOLD POLICY FUNCTIONS SEARCHING
    """
        solve_hh_decision_rule!(M::ModelBasic ; max_iter::Int = 700, rtol::Float64 = 1E-2, print_err::Bool = true, step_size::Float64 = 0.7, if_golden::Bool = true)

    solves household policy functions, using val func iter;
    returns a `HouseholdPolicy` instance.

    where `if_golden` controls if to use golden-section to precise-ize val func (if not, just use grid points of asset)
    """
    function solve_hh_decision_rule!(M::ModelBasic ; 
        max_iter::Int = 700, rtol::Float64 = 1E-2, print_err::Bool = true, step_size::Float64 = 0.7, if_golden::Bool = true)

        # init
        local vMat1 = zeros(M.Na, M.Nl) # old guess of val func
        local vMat2 = zeros(M.Na, M.Nl) # updated guess of val func
        local aOpt = zeros(M.Na, M.Nl) # opt asset, saving levels
        local cOpt = zeros(M.Na, M.Nl) # opt consumption, saving levels
        # init guess of val func
        for jl in 1:M.Nl, ja in 1:M.Na
            vMat1[ja,jl] = ufunc( get_c(M.lgrid[jl], M.agrid[ja], M.agrid[1], M.w), M.μ )
        end # for jl,ja

        # val func iter
        for iter in 1:max_iter
            # computes V(a,l)
            for jl in 1:M.Nl, ja in 1:M.Na
                # lambda
                local afunc_bellman(val_anext::Float64) = bellman( M.agrid[ja], jl, val_anext, vMat1, M )[1]
                # eval temp val func space for current (a,l), then find max
                # then, find grid max
                local tmploc = findmax([ afunc_bellman(M.agrid[x]) for x in 1:M.Na ])[2]

                # dispatching
                if if_golden  # if choose to precise-ize result
                    if (tmploc == 1) | (tmploc == M.Na)
                        aOpt[ja,jl] = M.agrid[tmploc]
                    else
                        # other wise, use golden section
                        local LLB = M.agrid[tmploc - 1]
                        local LB = M.agrid[tmploc]
                        local RB = M.agrid[tmploc + 1]
                        aOpt[ja,jl] = golden_max_3points(afunc_bellman, LLB, LB, RB, TOL = 1E-6)
                    end # if tmplocs
                else 
                    aOpt[ja,jl] = M.agrid[tmploc]  # save opt a grid
                end # if if_golden

                # eval at opt asset, save opt val func & consumption
                # NOTE: here we do not use afunc_bellman() because it only returns the 1st return
                vMat2[ja,jl], cOpt[ja,jl] = bellman( M.agrid[ja], jl, aOpt[ja,jl], vMat1, M )
            end # for jl, ja

            # check convergence
            local vFuncErr = findmax(abs.( vMat2 ./ ( vMat1 .+ eps() ) .- 1.0 ) )[1]
            print_err && println("- Round ",iter,", Rel Err: ",vFuncErr)

            if vFuncErr < rtol
                println("\t- HH Converged in Rel Err Meaning in round ",iter)
                break
            elseif iter == max_iter
                println("\t- HH Max Iter Reached")
            else
                vMat1 = (1.0 .- step_size) .* vMat1 .+ step_size .* vMat2  # update
            end # if isapprox()

        end # for iter

        # Modify model instance, add in HouseholdPolicy
        M.HHPolicy = HouseholdPolicy( M.agrid, M.lgrid, aOpt, cOpt, vMat1, vMat2 )
        
        return nothing
    end # solve_hh_decision_rule






# ======================= SOLVERS: STEADY STATE

    # ------------ SS SEARCH
    """
        solve_ss_gs!(M::ModelBasic, K_guess::Float64 ;
            max_iter_ss::Int = 500, max_iter_hh::Int = 500,
            rtol_ss::Float64 = 1E-3, rtol_hh::Float64 = 1E-3,
            step_size_ss::Float64 = 0.7, step_size_hh::Float64 = 0.7,
            print_err_ss::Bool = true, print_err_hh::Bool = false,
            if_golden::Bool = true, 
            seed_agg::Int = M.AGG_SEED, simu_size_agg::Int = M.AGG_SIMU_SIZE )

    solves steady state for given `ModelBasic`.s
    """
    function solve_ss_gs!(M::ModelBasic, K_guess::Float64 ;
        max_iter_ss::Int = 500, max_iter_hh::Int = 500,
        rtol_ss::Float64 = 1E-3, rtol_hh::Float64 = 1E-3,
        step_size_ss::Float64 = 0.7, step_size_hh::Float64 = 0.7,
        print_err_ss::Bool = true, print_err_hh::Bool = false,
        if_golden::Bool = true, 
        seed_agg::Int = M.AGG_SEED, simu_size_agg::Int = M.AGG_SIMU_SIZE )
        # ------------

        # init guess
        M.K = K_guess
        # init labor factor
        M.L = M.N * sum(M.π .* M.lgrid)
        
        # fixed-point iter
        for iter in 1:max_iter_ss
            # FIRM DEPT
            M.Y = (M.K)^(M.θ) * (M.L)^(1-M.θ);  # output
            M.r = (M.θ) * (M.K / M.L)^(1-M.θ) - M.δ;  # net interest rate
            M.w = (1-M.θ) * (M.K / M.L)^(M.θ);  # wage rate

            # SOCIAL SECURITY (PASSED)


            # HH DECISION MAKING
            print_err_ss && println("\t- Solving HH ...")
            solve_hh_decision_rule!(M, 
                max_iter = max_iter_hh, rtol = rtol_hh,
                print_err = print_err_hh, step_size = step_size_hh, if_golden = if_golden )

            # AGGREGATION
            Knew, M.C = agg_hh_simu(M, simu_size_agg, seed_agg ; a0 = M.K, drop_first = M.AGG_DROP_FIRST_SIZE)
            # ROUND SUMMARY
            if print_err_ss
                println("\t- agg K: ",Knew)
                println("\t- agg L: ",M.L)
                println("\t- agg Y: ",M.Y)
                println("\t- agg C: ",M.C)
                println("\t- agg S: ",M.Y - M.C)
                println("\t- r    : ",M.r)
                println("\t- w    : ",M.w)
            end # if print_err_ss

            # CHECK CONVERGENCE
            local IterErr = abs( Knew / ( M.K + eps() ) - 1.0 )
            print_err_ss && println("- Round ",iter," Rel Err: ",IterErr)

            if IterErr < rtol_ss
                println("- SS Converged in Rel Err Meaning")
                break
            elseif iter == max_iter_ss
                println("- SS Max Iter Reached")
            else
                (M.r < 0.001) && (Knew = 0.05)  # binding error fluctuation
                # update
                M.K = (1.0 - step_size_ss) * M.K + step_size_ss * Knew
            end # if
            
        end # for iter

        return nothing
    end # solve_ss_gs()

























end # src