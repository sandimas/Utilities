"""
references:
https://doi.org/10.1016/0370-1573(95)00074-7
https://www.cond-mat.de/events/correl12/manuscripts/jarrell.pdf
https://doi.org/10.1103/PhysRevE.94.023303
"""

using Statistics
using LinearAlgebra
using LogExpFunctions
using LinearSolve
using Dierckx # Spline functionality
using Printf

function calculate_spectral_function(G::AbstractMatrix{T},K::AbstractMatrix{T},m::AbstractVector{T};
                                    als::AbstractVector{T} = exp10.(range(9, stop=1, length=1+20*(9-1)))) where {T<:AbstractFloat}
    # Settings to ensure numerical stability
    W_ratio_max = 1e8
    svd_threshold = 1e-12 # drop very small singular values
    
    # estimate multivariate Gaussian distro from G
    n_bin = size(G,1)
    G_avg = Statistics.mean(G, dims=1)

    # calculate unitary matrix for diagonalizing covariance matrix
    svd_out = svd(G .- G_avg)
    sigma = svd_out.S
    Uc = svd_out.Vt
    W=(n_bin*(n_bin-1))./(sigma .* sigma)
    
    # deal with nearly singular covariance matrix by capping W
    W_cap = W_ratio_max * minimum(W)
    n_large = sum(maximum(W) > W_cap)
    if maximum(W) > W_cap
        for i = 1:n_bin
            if W[i,1] > W_cap
                W[i,1] = W_cap
            end
        end
    end

    # rotate K and G_avg
    Kp = Uc*K
    G_avg_p = zeros(T,n_bin)
    for i = 1:n_bin
        G_avg_p[i] = dot(view(Uc,i,:),G_avg)
    end

    # SVD of kernel
    svd_K = svd(Kp)
    V = svd_K.U
    Sigma = svd_K.S
    U = svd_K.Vt
    
    # drop singular values less than threshold
    mask = (Sigma/maximum(Sigma) .>= svd_threshold)

    # precalculate some values reused in looped calc_A_al()
    SigmaVt = (V[:,mask] .* Sigma[mask])
    M =  (W' .* SigmaVt)* SigmaVt'
    precalc = (U,SigmaVt,M)

    # arrays 
    us = zeros(size(als,1),size(M,1))
    chi2s = zeros(T,size(als))
    lnPs = zeros(T,size(als))
    dlnPs = zeros(T,size(als))
    # loop over alphas with calc_A_al()
    for (i, al) in enumerate(als)
        _, us[i,:], chi2s[i], lnPs[i], dlnPs[i] = calc_A_al(G_avg_p, W, Kp, m, al, precalc, view(us,i,:))
    end

    # Sorted alphas needed for Spline function
    order = sortperm(als)

    # Spline to fit, find max entropy
    fit = Spline1D(log.(als[order]),log.(chi2s[order]))
    k = derivative(fit,log.(als[order]),2)./(1 .+ derivative(fit,log.(als[order]),1).^2).^(1.5)
    max = findmax(k)[2]
    fit = Spline1D(log.(als[order]), dlnPs[order])
    al = als[order[max]]
    chi2 = chi2s[order[max]]
    
    A = m .* exp.(*(U',us[max, :]))
    print("\n", A,"\tA\n\n")
    dof = size(W,1) - n_large
    stat_str = @sprintf "alpha=%.3f\tchi2/dof=%.3f\tA.sum()=%.6f\n" al (chi2/dof) sum(A)
    print(stat_str)
  
    return A
end

function calc_A_al(G::AbstractArray{T}, W::AbstractArray{T}, K::AbstractArray{T}, m, al, precalc, u_init) where {T}
    mu_multiplier = 2.0 # Increase/decrease mu by this factor 
    mu_min = al/4.0 # minimum for non-zero mu
    mu_max = al*1e100 # maximum for non-zero mu
    step_max_accept = 0.5 # maximum size of accepted step
    step_drop_mu = 0.125 # decrease mu if step_size < this
    dQ_threshold = 1e-10
    max_small_dQ = 7 # stop if dQ/Q < dQ_threshold this many times in a row
    max_iter = 1234 # maximum number of iterations if conditions not met

    Q_new = 0
    n_bin = size(G,1)
    n_omega = size(m,1)
     
    # initial state
    U, SigmaVt, M = precalc
    u = u_init
    mu=al
    Id = 1.0*Matrix(I,size(M))
   
    pre_exp = zeros(size(m))
    for i = 1:size(m,1)
        pre_exp[i] = dot(u,view(U,:,i))
    end
    A = zeros(T,size(m))
    for i = 1:size(m,1)
        A[i] = m[i] * exp(pre_exp[i])
    end
    f = al*u + SigmaVt*((*(K,A)-G) .* W)
    Tmat = (U*(U' .* A))
    MT = Tmat*M'
    
    Q_old, S, chi2 = _Q(A, G, W, K, m, al)

    # search for max entryop
    small_dQ = 0
    for i=1:max_iter
        prob = LinearProblem((al+mu)*Id+MT,-f)
        du = (solve(prob).u)'
    
        step_size = (du*Tmat)⋅du
        for i = 1:size(m,1)
            pre_exp[i] = dot(view(U,:,i),u + du')
        end
        for i = 1:size(m,1)
            A[i] = m[i] * exp(pre_exp[i])
        end
        Q_new, S, chi2 = _Q(A, G, W, K, m, al)
        Q_ratio = Q_new/Q_old
        if step_size < step_max_accept && Q_ratio < 1000
            u += du'
            if abs(Q_ratio) < dQ_threshold
                small_dQ += 1
                if small_dQ == max_small_dQ
                    break
                end
            else
                small_dQ = 0
            end
            if step_size < step_drop_mu
                if mu > mu_min
                    mu = mu/mu_multiplier
                else
                    mu = 0.
                end
            end
            f = al*u + *(SigmaVt,(*(K,A)-G) .* W)
            Tmat =  (U*(U' .* A))'
            MT = *(M,Tmat)
            Q_old = Q_new
        else
            mu = clamp(mu*mu_multiplier,mu_min,mu_max)
        end
    end
    # calculate Z ∝ ln P(α|G,m)
    Z = zeros(T,n_bin,n_omega)
    for i =1:n_bin
        Z[i,1:4] = view(sqrt.(W)[:] .* K,i,:) .* sqrt.(A)
    end
    lam = (svd(Z)).S
    lam = lam .* lam
    lnP = 0.5 * sum(log.(al./(al .+ lam))) + Q_new
    S_vec = zeros(size(m))
    for i = 1:size(m,1)
        S_vec[i] = A[i] - m[i] - xlogy(A[i], A[i]/m[i])
    end
    S = sum(S_vec)
    dlnP = sum(lam./(al .+ lam))/(2 * al) + S 
    
    return A, u, chi2, lnP, dlnP
end

function _Q(A, G, W, K, m, al)
    S_vec = zeros(size(m))
    for i = 1:size(m,1)
        S_vec[i] = A[i] - m[i] - xlogy(A[i], A[i]/m[i])
    end
    S = sum(S_vec)
    KAG = *(K, A) - G
    chi2 = dot(KAG.*KAG, W)
    return al*S - 0.5*chi2, S, chi2
end

# Fermion kernel
function kernel_f(β::Real,τ::AbstractVector{T},ω::AbstractVector{T}) where {T}
    K = zeros(T,length(τ),length(ω))
    for oi = 1:length(ω)
        denom = 1.0 + exp(-β*ω[oi])
        for ti = 1:length(τ)
            K[ti,oi] = exp(-τ[ti]*ω[oi])/denom
        end
    end
    return K
end

function kernel_b(β::Real,τ::AbstractVector{T},ω::AbstractVector{T},sym=true) where {T}
    K = zeros(T,length(τ),length(ω))
    for oi = 1:length(ω)
        denom = 1.0 - exp(-β*ω[oi])
        if sym
            for ti = 1:length(τ)
                K[ti,oi] = ω* (exp(-τ[ti]*ω[oi]) + exp(-(β- τ[ti])*ω[oi]))/denom
            end
        else
            for ti = 1:length(τ)
                K[ti,oi] = ω* (exp(-τ[ti]*ω[oi]))/denom
            end
        end
    end
    return K
end

model = [0.1,.5,.3,.1]
beta = 1.0
tau = [1.0/3.0,2.0/3.0,1.0]
omegas = [-1.0, 0.0, 1.0, 2.0]
K_arr::AbstractArray = kernel_f(beta,tau,omegas)
myG = [.2 .1 .3; .01 .2 .3]
mix = calculate_spectral_function(myG,K_arr,model)