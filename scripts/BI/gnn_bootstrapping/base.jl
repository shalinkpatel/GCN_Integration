include("load.jl")
include("gnn.jl")
include("metrics.jl")

using JLD2

G = load_data()
base_model = build_base_gnn(G)

jldsave("data/base.model", model=base_model)
jldsave("data/base.results", 
    acc = acc(G, base_model), 
    loss = loss(G, base_model))