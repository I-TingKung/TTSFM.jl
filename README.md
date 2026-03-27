# TTSFM.jl

**TTSFM.jl** (Two-Tier Stochastic Frontier Model) is a robust and user-friendly Julia package designed for estimating Two-Tier Stochastic Frontier Models using the Maximum Simulated Likelihood Estimation (MSLE) method. 

It is inspired by the architecture of `SFrontiers.jl`, featuring a **Unified API** that abstracts mathematical complexity and delegates background optimization to dedicated, distribution-specific backends.
Ａ
---

## Features

- **Unified API Design**: `ttsfm_spec`, `ttsfm_method`, `ttsfm_init`, `ttsfm_opt`, and `ttsfm_fit`.
- **Multiple Error Distributions Supported**:
  - `HalfNormal`
  - `Exponential`
  - `Weibull`
  - `Pareto`
- **Smart Automatic Parameter Initialization**: Defaults automatically compute Ordinary Least Squares (OLS) starting values for the frontier to assist solver convergence.
- **Dynamic Parameter Mapping**: Variable names (e.g. `α, x1, Cu, z1, shape_u, σᵥ²`) in output tables are derived automatically from your matrix sizes and chosen distribution.

---

## Installation

You can install `TTSFM.jl` by cloning the repository to your environment or adding it as an active local path:

```julia
using Pkg
Pkg.add(path="path/to/TTSFM")
```

---

## Quick Start Example

Below is a minimal, complete example of generating specific initial values, specifying a Two-Tier model, and calling the solver. 

```julia
using TTSFM
using DataFrames, Random, Distributions

# Provide your dataset
# y_data : (N,) Array of Dependent Variable
# x_data : (N, K) Matrix of Frontier attributes (Do NOT include a constant column!)
# z_data : (N, Lu) Matrix of Inefficiency drivers for `u` (Do NOT include a constant column!)
# w_data : (N, Lw) Matrix of Inefficiency drivers for `w` (Do NOT include a constant column!)

# 1. Define Model Specification
myspec = ttsfm_spec(
    depvar=y_data,
    frontier=x_data,
    ineff_u_var=z_data,
    ineff_w_var=w_data,
    noise=:Normal,        # Statistical Noise term is strictly :Normal
    ineff_u=:Exponential, # Options: :Exponential, :HalfNormal, :Weibull, :Pareto
    ineff_w=:Exponential  # Note: ineff_u and ineff_w should currently match
)

# 2. Define Solvers Option (Optional) 
# Note: Draws are tied to Halton sequencing depth for quasi-random Monte-Carlo runs
mymeth = ttsfm_method(method=:MSLE, draws=2^10 - 1)

# 3. Create the Initialization Matrix (Optional)
# If you pass a completely empty spec here, TTSFM will intelligently guess initial values 
# by passing hcat(ones, X) into an OLS solver internally and initializing remaining scalars to 0.1.
myinit = ttsfm_init(
    spec=myspec
    # frontier = ... # Custom manual intercepts
)

# 4. Set Optim parameters (Optional)
# TTSFM utilizes a two-stage approach inside the MSLE workflow to improve convergence.
myopt = ttsfm_opt(
    warmstart_solver=NelderMead(),
    warmstart_opt=(iterations=10, g_abstol=1e-5),
    main_solver=Newton(),
    main_opt=(iterations=100, g_abstol=1e-5)
)

# 5. Fit the Model
# Printout table is beautifully stylized and fully automated for your given parameters!
result = ttsfm_fit(
    spec=myspec,
    method=mymeth,
    init=myinit,
    optim_options=myopt
)

# Call properties of result if needed!
# ex: result.coeff, result.stderror, result.t_stats
```

---

## Developer Guide

The **Unified API** resides within `src/TTSFM_API.jl` and `src/TTSFM.jl`. 

To extend `TTSFM` with another log-likelihood formulation, such as adding completely new probability distributions:
1. Append the math routine in the `src/sf_MSLE` folder directory.
2. Direct the backend router located in `src/sf_MSLE/MSLE_backend.jl` to interpret your newly assigned `:Symbol`.