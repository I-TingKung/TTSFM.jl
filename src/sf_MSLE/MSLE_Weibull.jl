function TTWB_msle(y, x, z, w, α, βx, Cu, βu, Cw, βw, shape_u, shape_w, Cv; draws=2^10-1)
    θᵤ = exp.(Cu .+ z * βu) .+ 1e-8
    θₘ = exp.(Cw .+ w * βw) .+ 1e-8
    σᵥ² = exp(Cv)
    σᵥ = sqrt(σᵥ²)
    N = length(y)
    ϵ = y .- α .- (x * βx)
    shape_u_pos = exp(shape_u) .+ 1e-8
    shape_w_pos = exp(shape_w) .+ 1e-8
    
    llike = Array{Real}(undef, N)
    u_halton_seq = collect(Halton(2, length=draws))
    w_halton_seq = collect(Halton(3, length=draws))

    @inbounds for i in 1:N
        u_samples = quantile.(Weibull(shape_u_pos, θᵤ[i]), u_halton_seq)
        w_samples = quantile.(Weibull(shape_w_pos, θₘ[i]), w_halton_seq)
        diff_samples = ϵ[i] .+ u_samples .- w_samples  
        llike[i] = log(mean(pdf_N.(diff_samples, σᵥ)) + eps()) 
    end
    return -sum(llike)
end
