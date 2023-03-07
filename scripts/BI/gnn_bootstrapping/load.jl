using Graphs, NPZ, GraphNeuralNetworks, JLD2

function load_data()
    x = npzread("data/x.npy")' |> Array
    y = npzread("data/y.npy")
    g = npzread("data/g.npy")

    G = SimpleDiGraph(length(y))
    foreach(r -> add_edge!(G, r[1] + 1, r[2] + 1), eachrow(g))
    return GNNGraph(G, ndata=(; x = x, y = y))
end

function load_model()
    return load("data/base.model", "model")
end