include("load.jl")
include("noise.jl")
include("metrics.jl")

using CairoMakie, AlgebraOfGraphics, DataFrames, DataFramesMeta, JLD2
CairoMakie.activate!(type="svg")

function eval_params(model :: GNNChain, G :: GNNGraph, λ, ρ)
    return acc(noisy_graph(G, λ, ρ), model)
end

base_model = load_model()
G = load_data()

λs = 1:10
ρs = 2:10
 
grid = Iterators.product(λs, ρs) |> collect
results = map(t -> eval_params(base_model, G, t[1], t[2]), grid)

@info results
jldsave("data/gnn_deg.jld", res=results)

heatmap(results)
