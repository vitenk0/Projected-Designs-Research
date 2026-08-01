using ITensors
using ITensorMPS

println("=== Test 1: CNOT(1→3), gap=1 ===")
sites = siteinds("Qubit", 4)
psi   = MPS(sites, "0")
psi   = apply(op("H", sites, 1), psi)
psi   = apply(op("CNOT", sites, 1, 3), psi)
s = siteinds(psi)
@show inner(MPS(s, ["0","0","0","0"]), psi)  # expect ≈ 0.707
@show inner(MPS(s, ["1","0","1","0"]), psi)  # expect ≈ 0.707
@show inner(MPS(s, ["0","0","1","0"]), psi)  # expect = 0.0

println("\n=== Test 2: CNOT(1→5), gap=3 (same as vertical bond in 4×4 grid) ===")
sites6 = siteinds("Qubit", 6)
psi6   = MPS(sites6, "0")
psi6   = apply(op("H", sites6, 1), psi6)
psi6   = apply(op("CNOT", sites6, 1, 5), psi6)
s6 = siteinds(psi6)
@show inner(MPS(s6, ["0","0","0","0","0","0"]), psi6)  # expect ≈ 0.707
@show inner(MPS(s6, ["1","0","0","0","1","0"]), psi6)  # expect ≈ 0.707
@show inner(MPS(s6, ["0","0","0","0","1","0"]), psi6)  # expect = 0.0

println("\n=== Test 3: debug single depth-5 circuit (L=4) ===")
using Random, Statistics, Printf

function random_clifford_t_gates(q1, q2; add_t=true, p_t=0.2)
    gates = []
    for q in (q1, q2)
        r = rand(0:3)
        r == 0 && push!(gates, ("H", q))
        r == 1 && push!(gates, ("S", q))
        if r == 2; push!(gates, ("H", q)); push!(gates, ("S", q)); end
    end
    push!(gates, rand() < 0.5 ? ("CNOT", (q1, q2)) : ("CNOT", (q2, q1)))
    if add_t
        for q in (q1, q2); rand() < p_t && push!(gates, ("T", q)); end
    end
    return gates
end

function build_circuit_2d(Lx, Ly, d; add_t=true, p_t=0.2, maxdim=64, cutoff=1e-10)
    N = Lx * Ly
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

Random.seed!(42)
psi = build_circuit_2d(4, 4, 5; add_t=true, p_t=0.15, maxdim=2048)
sites = siteinds(psi)
N = length(psi)
n_measured = 2
n_projected = N - n_measured

println("Checking p_x and amplitude for each x_bits:")
let total = 0.0
    for x_bits in Iterators.product(ntuple(_ -> (0, 1), n_measured)...)
        proj = copy(psi)
        for (i, bit) in enumerate(x_bits)
            s  = sites[i]
            Px = bit == 0 ? ITensor(ComplexF64[1 0; 0 0], s', dag(s)) :
                            ITensor(ComplexF64[0 0; 0 1], s', dag(s))
            proj[i] = noprime(proj[i] * Px)
        end
        p_x = real(inner(proj, proj))

        state_names = vcat([string(b) for b in x_bits], fill("0", n_projected))
        phi = MPS(sites, state_names)
        amp = inner(phi, psi)

        @printf("  x=%s  p_x=%.6f  |amp|=%.3e  |amp|²=%.3e\n",
            string(collect(x_bits)), p_x, abs(amp), abs(amp)^2)
        total += abs(amp)^4 / max(p_x, 1e-15)
    end
    result = (2.0^n_projected)^2 * total
    @printf("mps_small result: %.6e  (expected ≈ 2.0 for Haar)\n", result)
end
println("norm of psi = ", norm(psi), "  (should be 1.0)")
