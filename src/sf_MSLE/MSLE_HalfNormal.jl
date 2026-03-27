pdf_N(e, σ) = pdf(Normal(0, σ), e)

function TTNHN_msle(y, x, z, w, α, βx, Cu, βu, Cw, βw, Cv; draws=2^10-1)
    σᵤ = exp.(Cu .+ z * βu)  # parameterize parameter instead of variance
    σₘ = exp.(Cw .+ w * βw)
    σᵥ² = exp(Cv) # scalar
    σᵥ = sqrt(σᵥ²)
    N = length(y)
    ϵ = y .- α .- (x * βx)

    llike = Array{Real}(undef, N)
    u_halton_seq = collect(Halton(2, length=draws))
    w_halton_seq = collect(Halton(3, length=draws))
    
    @inbounds for i in 1:N
        u_samples = quantile.(truncated(Normal(0, σᵤ[i]), lower=0.0), u_halton_seq)
        w_samples = quantile.(truncated(Normal(0, σₘ[i]), lower=0.0), w_halton_seq)
        diff_samples = ϵ[i] .+ u_samples .- w_samples  
        llike[i] = log(mean(pdf_N.(diff_samples, σᵥ)) + eps())
    end
    return -sum(llike) 
end
