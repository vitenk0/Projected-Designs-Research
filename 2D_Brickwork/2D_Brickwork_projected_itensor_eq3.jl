# 2D Brickwork Clifford+T - Projected Ensemble Collision Probability
# ITensor MPS — eq.(3) single-outcome estimator (BACKUP)
# Formula: 2^N × |⟨0|ψ⟩|⁴ / p(x_B=0)²  with n_measured = N/2
#
# WARNING: high variance — dips to 10^-20 at certain depths due to
# the all-zero amplitude being systematically small for specific circuit layers.
# Use 2D_Brickwork_projected_itensor.jl (mps_small) for better results.

using ITensors
using ITensorMPS
using Random
using Statistics
using Printf
using Plots

function random_clifford_t_gates(q1, q2; add_t=true, p_t=0.2)
    gates = []
    for q in (q1, q2)
        r = rand(0:3)
        r == 0 && push!(gates, ("H", q))
        r == 1 && push!(gates, ("S", q))
        if r == 2
            push!(gates, ("H", q))
            push!(gates, ("S", q))
        end
    end
    push!(gates, rand() < 0.5 ? ("CNOT", (q1, q2)) : ("CNOT", (q2, q1)))
    if add_t
        for q in (q1, q2)
            rand() < p_t && push!(gates, ("T", q))
        end
    end
    return gates
end

function build_circuit_2d(Lx, Ly, d; add_t=true, p_t=0.2, maxdim=64, cutoff=1e-10)
    N         = Lx * Ly
    idx(x, y) = y * Lx + x + 1
    circuit = []
    for layer in 0:d-1
        direction = layer % 2
        offset    = (layer ÷ 2) % 2
        if direction == 0
            for y in 0:Ly-1, x in offset:2:(Lx-2)
                append!(circuit, random_clifford_t_gates(idx(x,y), idx(x+1,y); add_t, p_t))
            end
        else
            for x in 0:Lx-1, y in offset:2:(Ly-2)
                append!(circuit, random_clifford_t_gates(idx(x,y), idx(x,y+1); add_t, p_t))
            end
        end
    end
    sites = siteinds("Qubit", N)
    psi   = MPS(sites, "0")
    kw    = (maxdim=maxdim, cutoff=cutoff)
    for g in circuit
        name, site = g[1], g[2]
        G   = site isa Int ? op(name, sites, site) : op(name, sites, site...)
        psi = apply(G, psi; kw...)
    end
    return psi
end

function allzero_amplitude(psi)
    sites = siteinds(psi)
    return inner(MPS(sites, "0"), psi)
end

function marginal_zero_prob(psi, n_measured)
    sites = siteinds(psi)
    proj  = copy(psi)
    for i in 1:n_measured
        s       = sites[i]
        P0      = ITensor(ComplexF64[1 0; 0 0], s', dag(s))
        proj[i] = noprime(proj[i] * P0)
    end
    return real(inner(proj, proj))
end

function exact_projected_cp(psi)
    N          = length(psi)
    n_measured = N ÷ 2
    amp        = allzero_amplitude(psi)
    p_allzeros = abs(amp)^2
    p_x0       = marginal_zero_prob(psi, n_measured)
    denom      = p_x0^2
    (!isfinite(denom) || denom ≤ 0) && return NaN
    return (2.0^N) * p_allzeros^2 / denom
end

L             = 4
d_values      = 1:29
add_t         = true
max_bond      = 2048
samples_per_d = 20
output_dir    = "results/itensor_eq3"
mkpath(output_dir)

N           = L * L
n_measured  = N ÷ 2
n_projected = N - n_measured
avg_prob         = Float64[]
std_prob         = Float64[]
completed_depths = Int[]

for d in d_values
    vals = Float64[]
    for _ in 1:samples_per_d
        psi = build_circuit_2d(L, L, d; add_t=add_t, p_t=0.15, maxdim=max_bond)
        push!(vals, exact_projected_cp(psi))
    end
    valid     = filter(isfinite, vals)
    n_dropped = samples_per_d - length(valid)
    @printf("d=%d, N=%d, dropped %d/%d\n", d, N, n_dropped, samples_per_d)
    if isempty(valid)
        push!(avg_prob, NaN); push!(std_prob, NaN)
    else
        lv = log.(valid)
        push!(avg_prob, exp(mean(lv)))
        push!(std_prob, exp(std(lv)))
    end
    push!(completed_depths, d)
    lo  = avg_prob .- avg_prob ./ std_prob
    hi  = avg_prob .* std_prob .- avg_prob
    plt = plot(completed_depths, avg_prob; yerror=(lo,hi), marker=:circle,
        xlabel="depth", ylabel="2^(n_B) Σ_x p(x) Σ_y p(y|x)²",
        title="2D Brickwork: Projected CP eq.(3) backup (N=$(N), χ=$(max_bond))",
        label="N=$N", yscale=:log10, legend=:topright, grid=true)
    savefig(plt, joinpath(output_dir, "N$(N)_bond$(max_bond)_upto_d$(d).png"))
end
