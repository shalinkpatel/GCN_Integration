using Graphs, NPZ, GraphNeuralNetworks

function load_data()
    x = npzread("data/x.npy")' |> Array
    y = npzread("data/y.npy")
    g = npzread("data/g.npy")

    G = SimpleDiGraph(length(y))
    foreach(r -> add_edge!(G, r[1], r[2]), eachrow(g))
    return GNNGraph(G, ndata=(; x = x, y = y))
end