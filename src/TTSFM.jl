module TTSFM

using Random, Distributions, Optim, StatsFuns, DataFrames, CSV, LinearAlgebra, HaltonSequences, FLoops, GLM, ForwardDiff, Base.Threads, SpecialFunctions, ADTypes
using PrettyTables

# Export the unified API structures and wrapper functions
export ttsfm_spec, ttsfm_method, ttsfm_init, ttsfm_opt, ttsfm_fit

# Include API definitions
include("TTSFM_API.jl")

# Include Backend Dispatchers
include("sf_MSLE/MSLE_backend.jl")

end # module
