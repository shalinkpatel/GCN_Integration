using Flux, GraphNeuralNetworks, Graphs, ProgressBars
using Printf: @sprintf

function build_base_gnn(G :: GNNGraph)
    model = GNNChain(
        GCNConv(10 => 64, leakyrelu),
        GCNConv(64 => 64, leakyrelu),
        GCNConv(64 => 64, leakyrelu),
        Dense(64 => 2)
    )

    ps = Flux.params(model)
    opt = Adam(1f-4)

    loss(g :: GNNGraph) = Flux.logitcrossentropy(model(g, g.ndata.x), Flux.onehotbatch(g.ndata.y, 0:1))

    iter = ProgressBar(1:10_000)
    for epoch ∈ iter
        grad = gradient(ps) do 
            train = loss(G)
            return train
        end
        Flux.Optimise.update!(opt, ps, grad)

        set_description(iter, @sprintf("Loss: %.5f, Acc: %.5f", train_loss = loss(G), acc = sum(Flux.onecold(softmax(model(G, G.ndata.x)), 0:1) .== G.ndata.y) / length(G.ndata.y)) |> string)
    end

    return model
end