using Statistics
using LinearAlgebra
using LogExpFunctions
using LinearSolve
using Dierckx
using Printf

function calculate_spectral_function(G::AbstractMatrix{T},K::AbstractMatrix{T},m::AbstractVector{T};
                                    als::AbstractVector{T} = exp10.(range(9, stop=1, length=1+20*(9-1)))) where {T}
    #                                  ) where {T<:AbstractFloat}
    W_ratio_max = 1e8
    svd_threshold = 1e-12
    n_bin = size(G,1)
    # print(size(G),"x\n")
    G_avg = Statistics.mean(G, dims=1)
    
    svd_out = svd(G .- G_avg)
    sigma = svd_out.S
    Uc = svd_out.Vt
    
    W=(n_bin*(n_bin-1))./(sigma .* sigma)
    # print(size(W),W,"W\n")
    W_cap = W_ratio_max * minimum(W)
    n_large = sum(maximum(W) > W_cap)
    if maximum(W) > W_cap
        for i = 1:n_bin
            if W[i,1] > W_cap
                W[i,1] = W_cap
            end
        end

    end
    Kp = *(Uc,K)
    # print(size(G_avg),G_avg,"gavg\n")
    # print(size(Uc),Uc,"Uc\n")
    
    G_avg_p = zeros(T,n_bin)
    # print("here\n")
    for i = 1:n_bin
        # print(view(Uc,i,:),i,"Uc\n")
        G_avg_p[i] = dot(view(Uc,i,:),G_avg)
    end
    svd_K = svd(Kp)
    V = svd_K.U
    Sigma = svd_K.S
    U = svd_K.Vt
    # mask = Array{Bool}(n_bin)
    mask = (Sigma/maximum(Sigma) .>= svd_threshold)
    # print(size(mask),mask,"mask\n")
    SigmaVt = (V[:,mask] .* Sigma[mask])
    # M_tmp = zeros(T,n_bin,n_bin)
    # print(SigmaVt,"svt\n")
    # print(size(SigmaVt),size(W),"size\n")
    M_tmp = transpose(transpose(SigmaVt) .* W)
    M =  *(M_tmp,transpose(SigmaVt))
    precalc = (U,SigmaVt,M)
    # print(size(als,1),"als\n")
    us = zeros(size(als,1),size(M,1))
    # print(size(us),size(us[1]),"us\n\n")
    chi2s = zeros(T,size(als))
    lnPs = zeros(T,size(als))
    dlnPs = zeros(T,size(als))
    for (i, al) in enumerate(als)
        if i==1
            _, us[i,:], chi2s[i], lnPs[i], dlnPs[i] = calc_A_al(G_avg_p, W, Kp, m, al, precalc, view(us,i,:))
        else
            _, us[i,:], chi2s[i], lnPs[i], dlnPs[i] = calc_A_al(G_avg_p, W, Kp, m, al, precalc, view(us,i,:))
        end 
    end
    order = sortperm(als)
    # BT
    fit = Spline1D(log.(als[order]),log.(chi2s[order]))
    k = derivative(fit,log.(als[order]),2)./(1 .+ derivative(fit,log.(als[order]),1).^2).^(1.5)
    max = findmax(k)[2]
    fit = Spline1D(log.(als[order]), dlnPs[order])
    al = als[order[max]]
    chi2 = chi2s[order[max]]
    
    A = m .* exp.(*(U',us[max, :]))
    print(A,"A\n\n")
    dof = size(W,1) - n_large
    err_str = @sprintf "alpha=%.3f\tchi2/dof=%.3f\tA.sum()=%.6f\n" al (chi2/dof) sum(A)
    print(err_str)
    # print(f"alpha={al:.3f}\tchi2/dof={chi2/dof:.3f}\tA.sum()={A.sum():6f}")

    return A
end

function calc_A_al(G::AbstractArray{T}, W::AbstractArray{T}, K::AbstractArray{T}, m, al, precalc, u_init) where {T}
    svd_threshold = 1e-12
    mu_multiplier = 2.0
    mu_min = al/4.0
    mu_max = al*1e100
    step_max_accept = 0.5
    step_drop_mu = 0.125
    dQ_threshold = 1e-10
    max_small_dQ = 7
    max_iter = 1234
    Q_new = 0
    U, SigmaVt, M = precalc
    u = u_init
    mu=al
    n_bin = size(G,1)
    n_omega = size(m,1)
    Id = 1.0*Matrix(I,size(M))
    # print(size(U),size(u),"U\n")
    pre_exp = zeros(size(m))
    for i = 1:size(m,1)
        pre_exp[i] = dot(u,view(U,:,i))
    end
    A = zeros(T,size(m))
    for i = 1:size(m,1)
        A[i] = m[i] * exp(pre_exp[i])
    end
    f = al*u + *(SigmaVt,(*(K,A)-G) .* W)
    # print(size(A),size(U),"sizes\n")
    Tmat =  *(transpose(transpose(U) .* A),transpose(U))
    MT = *(M,Tmat)
    # print(size(MT),MT,"MT\n")
    Q_old, S, chi2 = _Q(A, G, W, K, m, al)
    small_dQ = 0
    for i=1:max_iter
        prob = LinearProblem((al+mu)*Id+MT,-f)
        du = transpose(solve(prob).u)
        # print(du,"du\n")
        step_size = dot(*(du,Tmat),du)
        # print(step_size,"step\n")
        # print(size(u),size(du),size(U))
        for i = 1:size(m,1)
            pre_exp[i] = dot(view(U,:,i),u + transpose(du))
        end
        for i = 1:size(m,1)
            A[i] = m[i] * exp(pre_exp[i])
        end
        Q_new, S, chi2 = _Q(A, G, W, K, m, al)
        Q_ratio = Q_new/Q_old
        if step_size < step_max_accept && Q_ratio < 1000
            u += transpose(du)
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
            Tmat =  *(transpose(transpose(U) .* A),transpose(U))
            MT = *(M,Tmat)
            Q_old = Q_new
        else
            mu = clamp(mu*mu_multiplier,mu_min,mu_max)
        end
    end
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
    # print(dlnP,"dlnp\n")
    # print(u,"u\n")
    # exit()
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


model = [0.1,.5,.3,.1]
beta = 1.0
tau = [1.0/3.0,2.0/3.0,1.0]
omegas = [-1.0, 0.0, 1.0, 2.0]
K_arr::AbstractArray = kernel_f(beta,tau,omegas)
myG = [.2 .1 .3; .01 .2 .3]
mix = calculate_spectral_function(myG,K_arr,model)