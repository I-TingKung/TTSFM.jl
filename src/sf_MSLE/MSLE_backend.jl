# Dispatcher for MSLE
include("MSLE_HalfNormal.jl")
include("MSLE_Exponential.jl")
include("MSLE_Weibull.jl")
include("MSLE_Pareto.jl")

# Helper to fix positive definite nature of Hessian
function make_positive_definite(H; ϵ=1e-6)
    eigenvals, eigenvecs = eigen(Symmetric(H))
    corrected_eigenvals = max.(eigenvals, ϵ)
    return eigenvecs * Diagonal(corrected_eigenvals) * eigenvecs'
end

function compute_var_cov_matrix(H; λ=1e-6)
    local var_cov_matrix
    try
        var_cov_matrix = inv(H)
    catch e1
        try
            L = cholesky(H)
            var_cov_matrix = L \ I
        catch e2
            try
                H_reg = H + λ * I
                L = cholesky(H_reg)
                var_cov_matrix = L \ I
            catch e3
                try
                    var_cov_matrix = pinv(H)
                catch e4
                    error("All methods failed. Hessian too ill-conditioned.")
                end
            end
        end
    end
    return var_cov_matrix
end

function _fit_msle(spec::ttsfm_spec, method::ttsfm_method, init::ttsfm_init, optim_options::ttsfm_opt)
    y = spec.depvar
    x = spec.frontier
    z = spec.ineff_u_var
    w = spec.ineff_w_var
    
    # 1. Routing to the proper likelihood function and setting up initial variables
    local target_func
    local n_vars
    local K = size(x, 2)
    local Lu = size(z, 2)
    local Lw = size(w, 2)
    
    if spec.ineff_u == :HalfNormal && spec.ineff_w == :HalfNormal
        # α, βx*K, Cu, βu*Lu, Cw, βw*Lw, Cv
        n_vars = 1 + K + 1 + Lu + 1 + Lw + 1
        target_func = (vars) -> TTNHN_msle(y, x, z, w, vars[1], vars[2:1+K], vars[2+K], vars[3+K:2+K+Lu], vars[3+K+Lu], vars[4+K+Lu:3+K+Lu+Lw], vars[end]; draws=method.draws)
    elseif spec.ineff_u == :Exponential && spec.ineff_w == :Exponential
        n_vars = 1 + K + 1 + Lu + 1 + Lw + 1
        target_func = (vars) -> TTEX_msle(y, x, z, w, vars[1], vars[2:1+K], vars[2+K], vars[3+K:2+K+Lu], vars[3+K+Lu], vars[4+K+Lu:3+K+Lu+Lw], vars[end]; draws=method.draws)
    elseif spec.ineff_u == :Weibull && spec.ineff_w == :Weibull
        # shape_u, shape_w at end-2, end-1
        n_vars = 1 + K + 1 + Lu + 1 + Lw + 2 + 1
        target_func = (vars) -> TTWB_msle(y, x, z, w, vars[1], vars[2:1+K], vars[2+K], vars[3+K:2+K+Lu], vars[3+K+Lu], vars[4+K+Lu:3+K+Lu+Lw], vars[end-2], vars[end-1], vars[end]; draws=method.draws)
    elseif spec.ineff_u == :Pareto && spec.ineff_w == :Pareto
        n_vars = 1 + K + 1 + Lu + 1 + Lw + 2 + 1
        target_func = (vars) -> TTPT_msle(y, x, z, w, vars[1], vars[2:1+K], vars[2+K], vars[3+K:2+K+Lu], vars[3+K+Lu], vars[4+K+Lu:3+K+Lu+Lw], vars[end-2], vars[end-1], vars[end]; draws=method.draws)
    else
        error("Combination of ineff_u=$(spec.ineff_u) and ineff_w=$(spec.ineff_w) is not currently supported in MSLE backend.")
    end

    # Use specified initial values or default
    init_vars = Float64[]
    
    # 1. Frontier (size: 1 + K)
    if init.frontier === nothing
        # auto OLS
        X_with_intercept = hcat(ones(length(y)), x)
        append!(init_vars, (X_with_intercept \ y))
    else
        append!(init_vars, init.frontier)
    end

    # 2. Cu (size: 1)
    push!(init_vars, init.Cu === nothing ? 0.1 : init.Cu)

    # 3. βu (size: Lu)
    if init.βu === nothing
        append!(init_vars, 0.1 * ones(Lu))
    else
        append!(init_vars, init.βu)
    end

    # 4. Cw (size: 1)
    push!(init_vars, init.Cw === nothing ? 0.1 : init.Cw)

    # 5. βw (size: Lw)
    if init.βw === nothing
        append!(init_vars, 0.1 * ones(Lw))
    else
        append!(init_vars, init.βw)
    end

    # 6. Optional shape_u, shape_w
    if spec.ineff_u in (:Weibull, :Pareto) && spec.ineff_w in (:Weibull, :Pareto)
        push!(init_vars, init.shape_u === nothing ? 0.1 : init.shape_u)
        push!(init_vars, init.shape_w === nothing ? 0.1 : init.shape_w)
    end

    # 7. Cv
    push!(init_vars, init.Cv === nothing ? 0.1 : init.Cv)

    # 2. Setup differentiable function
    func = TwiceDifferentiable(target_func, init_vars; autodiff = AutoForwardDiff())

    # 3. Optimisation
    res_nm = optimize(target_func, init_vars, optim_options.warmstart_solver, Optim.Options(; optim_options.warmstart_opt...))
    init_newton = Optim.minimizer(res_nm)
    res_nt = optimize(func, init_newton, optim_options.main_solver, Optim.Options(; optim_options.main_opt...))

    # 4. Results extraction
    coeff = Optim.minimizer(res_nt)
    res_nt_hessian = ForwardDiff.hessian(func.f, coeff)

    if !issymmetric(res_nt_hessian)
        res_nt_hessian = 0.5 * (res_nt_hessian + res_nt_hessian')
    end

    if !isposdef(res_nt_hessian)
        res_nt_hessian = make_positive_definite(res_nt_hessian)
    end

    TT_coeff = deepcopy(coeff)
    # Exponentiate parameters depending on the distribution
    if spec.ineff_u in (:Weibull, :Pareto) && spec.ineff_w in (:Weibull, :Pareto)
        TT_coeff[end-2:end] = exp.(TT_coeff[end-2:end]) # shape_u, shape_w, sigma_v_sq
    else
        TT_coeff[end] = exp.(TT_coeff[end]) # only sigma_v_sq
    end

    _Hessian = Optim.hessian!(func, coeff)
    var_cov_matrix = compute_var_cov_matrix(_Hessian)
    stderror = sqrt.(diag(var_cov_matrix))
    
    if spec.ineff_u in (:Weibull, :Pareto) && spec.ineff_w in (:Weibull, :Pareto)
        stderror[end-2:end] = TT_coeff[end-2:end] .* stderror[end-2:end]
    else
        stderror[end] = TT_coeff[end] .* stderror[end]
    end
    t_stats = TT_coeff ./ stderror

    table = hcat(TT_coeff, stderror, t_stats)
    
    # Generate Rownames dynamically if not provided
    if isempty(spec.rownames) || length(spec.rownames) != n_vars
        auto_rownames = String["α"]
        append!(auto_rownames, ["x$i" for i in 1:K])
        push!(auto_rownames, "Cu")
        append!(auto_rownames, ["z$i" for i in 1:Lu])
        push!(auto_rownames, "Cw")
        append!(auto_rownames, ["w$i" for i in 1:Lw])
        if spec.ineff_u in (:Weibull, :Pareto) && spec.ineff_w in (:Weibull, :Pareto)
            push!(auto_rownames, "shape_u", "shape_w")
        end
        push!(auto_rownames, "σᵥ²")
    else
        auto_rownames = spec.rownames
    end

    # Render table
    df = DataFrame(table, ["Coefficient", "StdError", "TStatistics"])
    insertcols!(df, 1, :Variable => auto_rownames)
    pretty_table(df; title="TTSFM Estimation Table: $(spec.ineff_u) / $(spec.ineff_w)")
    
    # Pack result
    return (
        coeff = TT_coeff,
        stderror = stderror,
        t_stats = t_stats,
        optim_result = res_nt,
        hessian = res_nt_hessian
    )
end
