include("load.jl")

using Graphs, Distributions, Lazy, StatsBase

function noisy_graph(G :: GNNGraph, λ, ρ)
    ei = edge_index(G)

    g = SimpleDiGraph(G.ndata.y |> length)
    for (s, t) ∈ zip(ei[1], ei[2])
        add_edge!(g, s, t)
    end

    for n ∈ 1:ne(g)
        e = rand(Poisson(λ))
        candidates = @>> neighborhood_dists(g, 570, ρ) filter(t -> t[2] > 1)
        targets = sample(map(t -> t[1], candidates), Weights(map(t -> 1 / t[2], candidates)), e)
        foreach(t -> add_edge!(g, n, t), targets)
    end

    return GNNGraph(g, ndata=(; x = G.ndata.x, y = G.ndata.y))
end