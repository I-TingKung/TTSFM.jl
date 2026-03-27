# Unified API Structures

Base.@kwdef struct ttsfm_spec
    depvar::Vector{Float64}
    frontier::Matrix{Float64}
    ineff_u_var::Matrix{Float64}
    ineff_w_var::Matrix{Float64}
    noise::Symbol = :Normal
    ineff_u::Symbol   # e.g., :HalfNormal, :Exponential, :Weibull, :Pareto
    ineff_w::Symbol   # e.g., :HalfNormal, :Exponential, :Weibull, :Pareto
    rownames::Vector{String} = [] # Optional labels for the output table
end

Base.@kwdef struct ttsfm_method
    method::Symbol = :MSLE
    draws::Int = 2^10 - 1
end

Base.@kwdef struct ttsfm_init
    spec::ttsfm_spec
    frontier::Union{Vector{Float64}, Nothing} = nothing # α + βx
    Cu::Union{Float64, Nothing} = nothing
    βu::Union{Vector{Float64}, Nothing} = nothing
    Cw::Union{Float64, Nothing} = nothing
    βw::Union{Vector{Float64}, Nothing} = nothing
    Cv::Union{Float64, Nothing} = nothing
    shape_u::Union{Float64, Nothing} = nothing
    shape_w::Union{Float64, Nothing} = nothing
end

Base.@kwdef struct ttsfm_opt
    warmstart_solver::Any = NelderMead()
    warmstart_opt::NamedTuple = (iterations=10, g_abstol=1e-3)
    main_solver::Any = Newton()
    main_opt::NamedTuple = (iterations=50, g_abstol=1e-5)
end

# Main entry point to fit the model
function ttsfm_fit(; spec::ttsfm_spec, method::ttsfm_method, init::ttsfm_init, optim_options::ttsfm_opt)
    # Check method
    if method.method == :MSLE
        # Dispatch to MSLE backend
        return _fit_msle(spec, method, init, optim_options)
    else
        error("Method $(method.method) is not implemented.")
    end
end
