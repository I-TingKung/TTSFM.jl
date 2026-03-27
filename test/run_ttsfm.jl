## 1. Environment Setup
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

## 2. Include TTSFM functionality
include("../src/TTSFM.jl")
using .TTSFM
using Random, DataFrames, Optim, Distributions

## 3. Data Simulation
Random.seed!(42)
N = 500

# True parameters
α_true = 1.5
β_true = [0.5, -0.3]
# x1, x2
x_data = randn(N, 2)

# Z variables for u (HalfNormal or Exponential standard dev parameters)
z_data = randn(N, 1)
Cu_true = -0.5
βu_true = [0.2]
σᵤ = exp.(Cu_true .+ z_data * βu_true)

# W variables for w
w_data = randn(N, 1)
Cw_true = -0.8
βw_true = [0.4]
σₘ = exp.(Cw_true .+ w_data * βw_true)

σᵥ_true = 0.5

# Generate errors mapping TTEX model (Exponential)
# We will simulate data under the Exponential assumption for simplicity
u_true = rand.(Exponential.(σᵤ))
w_true = rand.(Exponential.(σₘ))
v_true = randn(N) .* σᵥ_true

y_data = α_true .+ x_data * β_true .+ v_true .+ u_true .- w_true

# (You may change the distribution of u and w to HalfNormal by changing the code below)
# u_true = rand.(HalfNormal.(σᵤ))
# w_true = rand.(HalfNormal.(σₘ))
# (So do Pareto and Weibull)

# (You may also load the dataset from a CSV file by changing the code below)
# df = CSV.read("data.csv", DataFrame)
# y_data = df.y
# x_data = df.x
# z_data = df.z
# w_data = df.w

## 4. Estimation using the Unified API

# Set up the model spec
myspec = ttsfm_spec(
    depvar=y_data,
    frontier=x_data,
    ineff_u_var=z_data,
    ineff_w_var=w_data,
    noise=:Normal,    # Currently only :Normal is available.
    ineff_u=:Exponential, # :Exponential, :HalfNormal, :Weibull, :Pareto
    ineff_w=:Exponential  # :Exponential, :HalfNormal, :Weibull, :Pareto
    # Currently please choose the same ones.
)

println("Estimating TTSFM using $(myspec.ineff_u) / $(myspec.ineff_w) Specification...")

# Choose MSLE method
mymeth = ttsfm_method(method=:MSLE, draws=2^10 - 1)

# Because we have 2 X vars, 1 Z var, 1 W var + constants = 1+2+1+1+1+1+1 = 8 parameters
# Defaults will automatically run OLS for the frontier, and populate Cu, βu, Cw, βw, Cv with 0.1.
myinit = ttsfm_init(
    spec=myspec
    # frontier=ols_coeffs, # default
    # Cu=0.1,              # default
    # βu=[0.1],            # default
    # Cw=0.1,              # default
    # βw=[0.1],            # default
    # Cv=0.1               # default
)

myopt = ttsfm_opt(
    warmstart_solver=NelderMead(),
    warmstart_opt=(iterations=10, g_abstol=1e-3),
    main_solver=Newton(), # BFGS is also available.
    main_opt=(iterations=100, g_abstol=1e-5)
)

result = ttsfm_fit(
    spec=myspec,
    method=mymeth,
    init=myinit,
    optim_options=myopt
)

println("Test run completed successfully!")