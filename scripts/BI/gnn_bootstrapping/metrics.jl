using GraphNeuralNetworks, Flux

acc(G :: GNNGraph, model :: GNNChain) = sum(Flux.onecold(softmax(model(G, G.ndata.x)), 0:1) .== G.ndata.y) / length(G.ndata.y)
loss(G :: GNNGraph, model :: GNNChain) = Flux.logitcrossentropy(model(G, G.ndata.x), Flux.onehotbatch(G.ndata.y, 0:1))

