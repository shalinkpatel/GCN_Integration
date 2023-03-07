include("load.jl")
include("noise.jl")

G = load_data()
λ = 2.5
ρ = 3

Ḡ = noisy_graph(G, λ, ρ)

open("data/noisy.data", "w") do f
    for (s, d) ∈ zip(Ḡ.graph[1], Ḡ.graph[2])
        write(f, "$(s-1),$(d-1)\n")
    end
end;
