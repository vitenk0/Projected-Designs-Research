# 2D Brickwork Clifford+T - Projected Ensemble Collision Probability
# ITensor MPS — Born-sampled estimator, half/half bipartition
#
# Bipartition: B = first N/2 sites (rows y=0..Ly/2-1 of the lattice, i.e. the
# bottom half), A = remaining N/2 sites. B is measured first to produce the
# projected ensemble on A; the collision probability then corresponds to a
# computational-basis measurement of A.
#
# Quantity: 2^(n_A) · Σ_{x∈{0,1}^{n_B}} p(x) · Σ_y p(y|x)²
#
# Estimator: sample full bitstrings z=(x,y) ~ p(z) = |⟨z|ψ⟩|² by sweeping the
# MPS. Then
#     Σ_x p(x) Σ_y p(y|x)²  =  E_{z~p}[ p(y|x) ]  =  E_{z~p}[ p(z)/p(x) ],
# and both p(z) and p(x) come for free during the sequential sampling sweep
# (product of site-wise conditionals over all sites / over the B sites only).
# Unbiased and low-variance; exact per-sample for stabilizer states.
#
# Haar reference for the projected state on A:  2·2^(n_A)/(2^(n_A)+1) ≈ 2

using ITensors
using ITensorMPS
using Random
using Statistics
using Printf
using Plots

# ── Random Clifford+T block → gate tuples ────────────────────────────────────

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
        # r == 3: identity
    end
    push!(gates, rand() < 0.5 ? ("CNOT", (q1, q2)) : ("CNOT", (q2, q1)))
    if add_t
        for q in (q1, q2)
            rand() < p_t && push!(gates, ("T", q))
        end
    end
    return gates
end

# ── 2D brickwork circuit ──────────────────────────────────────────────────────

function build_circuit_2d(Lx, Ly, d; add_t=true, p_t=0.2, maxdim=64, cutoff=1e-10)
    N         = Lx * Ly
    idx(x, y) = y * Lx + x + 1  # 0-indexed (x,y) → 1-indexed site

    circuit = []
    for layer in 0:d-1
        direction = layer % 2
        offset    = (layer ÷ 2) % 2
        if direction == 0  # horizontal bonds
            for y in 0:Ly-1, x in offset:2:(Lx-2)
                append!(circuit, random_clifford_t_gates(idx(x,y), idx(x+1,y); add_t, p_t))
            end
        else               # vertical bonds
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

# ── Sequential Born sampling with probability tracking ───────────────────────
#
# Adapted from ITensorMPS.sample: sweeps the MPS site by site, drawing each
# bit from its conditional distribution. The product of the site-wise
# conditionals over the first n_measured sites is p(x); over all sites, p(z).
# phi must be normalized with orthogonality center at site 1.

function sample_z_probs(phi::MPS, n_measured::Int)
    N    = length(phi)
    bits = Vector{Int}(undef, N)
    p_z  = 1.0
    p_x  = 1.0
    A    = phi[1]
    for j in 1:N
        s     = siteind(phi, j)
        r     = rand()
        pdisc = 0.0
        n     = 1
        An    = ITensor()
        pn    = 0.0
        while true
            projn        = ITensor(s)
            projn[s => n] = 1.0
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            pdisc += pn
            (r < pdisc || n == 2) && break
            n += 1
        end
        bits[j] = n - 1
        p_z    *= pn
        j <= n_measured && (p_x *= pn)
        pn <= 0 && return bits, 0.0, p_x   # numerically dead branch
        if j < N
            A  = phi[j + 1] * An
            A *= (1.0 / sqrt(pn))
        end
    end
    return bits, p_z, p_x
end

# ── Born-sampled estimator ────────────────────────────────────────────────────

function projected_cp_born(psi; n_measured, n_z_samples=100)
    N           = length(psi)
    n_projected = N - n_measured

    phi = orthogonalize(psi, 1)
    normalize!(phi)

    total  = 0.0
    n_used = 0
    for _ in 1:n_z_samples
        _, p_z, p_x = sample_z_probs(phi, n_measured)
        p_x <= 1e-15 && continue
        total  += p_z / p_x        # = p(y|x)
        n_used += 1
    end

    n_used == 0 && return NaN
    val = 2.0^n_projected * total / n_used
    return isfinite(val) ? val : NaN
end

# ── Vary-depth sweep (L=4, N=16, d=1..29) ────────────────────────────────────

L             = 5
d_values      = 1:29
add_t         = true
max_bond      = 2048
samples_per_d = 20
n_z_samples   = 100
output_dir    = "results/itensor"
mkpath(output_dir)

N           = L * L
n_measured  = N ÷ 2        # half the system measured out (bottom two rows)
n_projected = N - n_measured

avg_prob         = Float64[]
sem_prob         = Float64[]
completed_depths = Int[]

function make_plot(depths, avg, sem)
    # Keep the lower error bar strictly positive so log-scale rendering works
    lo  = min.(sem, avg .* (1 - 1e-6))
    plt = plot(depths, avg;
        yerror = (lo, sem),
        marker = :circle,
        xlabel = "depth",
        ylabel = "2^(n_A) Σ_x p(x) Σ_y p(y|x)²",
        title  = "2D Brickwork: Projected CP (Born-sampled, N=$(N), n_meas=$(n_measured), χ=$(max_bond))",
        label  = "N=$N",
        yscale = :log10,
        legend = :topright,
        grid   = true)
    return plt
end

for d in d_values
    vals = Float64[]
    for _ in 1:samples_per_d
        psi = build_circuit_2d(L, L, d; add_t=add_t, p_t=0.15, maxdim=max_bond)
        push!(vals, projected_cp_born(psi; n_measured=n_measured,
                                           n_z_samples=n_z_samples))
    end

    valid     = filter(v -> isfinite(v) && v > 0, vals)
    n_dropped = samples_per_d - length(valid)
    @printf("d=%d, N=%d, n_measured=%d, n_projected=%d, dropped %d/%d\n",
        d, N, n_measured, n_projected, n_dropped, samples_per_d)

    if isempty(valid)
        push!(avg_prob, NaN); push!(sem_prob, NaN)
    else
        # Arithmetic mean over circuit instances (the ensemble average E_ψ),
        # with standard error of the mean for the error bars
        push!(avg_prob, mean(valid))
        push!(sem_prob, std(valid) / sqrt(length(valid)))
    end
    push!(completed_depths, d)

    plt = make_plot(completed_depths, avg_prob, sem_prob)
    savefig(plt, joinpath(output_dir,
        "N$(N)_nmeas$(n_measured)_bond$(max_bond)_upto_d$(d).png"))
    @printf("  saved upto_d%d.png\n", d)
end

display(make_plot(completed_depths, avg_prob, sem_prob))