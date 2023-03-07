using Graphs, Turing, PyCall, Distributions

nx = pyimport("networkx")
tg = pyimport("torch_geometric")
tch = pyimport("torch")

@model function rw_model(edge_list, N, p, n₀, X, y, model)
    ne = size(edge_list, 2)
    τᵢ = tzeros(Bool, ne)
    τᵢ ~ filldist(Bernoulli(p), ne)
    G = SimpleDiGraph(N)

    mapping = Dict{Edge, Int}()

    for (i,edge) ∈ enumerate(eachcol(edge_list))
        add_edge!(G, edge[1], edge[2])
        mapping[Edge(edge[1], edge[2])] = i
    end

    nodes = Set{Int}()
    possible = Set{Edge}()
    added = Set{Edge}()
    visited = Set{Edge}()
    
    push!(nodes, n₀)
    for edge ∈ incident(G, nodes)
        push!(possible, edge)
    end

    while length(possible) != 0
        consider = rand(possible)
        delete!(possible, consider)
        push!(visited, consider)

        idx = mapping[consider]
        if τᵢ[idx]
            push!(added, consider)
            push!(nodes, consider.src)
            push!(nodes, consider.dst)
        end

        for edge ∈ incident(G, nodes)
            if !(edge ∈ added) && !(edge ∈ visited)
                push!(possible, edge)
            end
        end
    end

    μ = model([map(x -> x.src, collect(added)), map(x -> x.dst, collect(added))])
    y = rand(Categorical(y/sum(y)))
    y ~ Categorical(μ/sum(μ))
end

function incident(G, ns)
    return filter(e -> e.dst ∈ ns, collect(edges(G)))
end
