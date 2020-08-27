using DrWatson
@quickactivate "GCN_HM_GRN-Integration"
using Turing, PyCall 

pushfirst!(PyVector(pyimport("sys")."path"), "/gpfs_home/spate116/singhlab/GCN_Integration/notebooks/VI/");
train_model = pyimport("gcn").train_model
run_model = pyimport("gcn").run_model
extract_subgraph = pyimport("gcn").extract_subgraph
pyimport("sys")."stdout" = PyTextIO(stdout)
pyimport("sys")."stderr" = PyTextIO(stderr)

using Zygote
Zygote.@adjoint function pycall(f, x...; kw...)
    x = map(py, x)
    y = pycall(f, x...; kw...)
    y.detach().numpy(), function (ȳ)
        y.backward(gradient = py(ȳ))
        (nothing, map(x->x.grad.numpy(), x)...)
    end
end

node = 10
model, x, y, edge_index = train_model();
edge_index, node_idx = extract_subgraph(node, 2, edge_index)
edge_mask = repeat([1], get(edge_index."shape", 1));

using Flux
Turing.setadbackend(:zygote)
@model graph_mask(y) = begin
    N = get(edge_index."shape", 1)
    edge_mask = Vector{Int}(undef, N)
    for i ∈ 1:N
        edge_mask[i] ~ Bernoulli(0.1)
    end
    pred = run_model(edge_mask, edge_index, model, node_idx)
    y = y/sum(y)
    y ~ MvNormal(pred/sum(pred), repeat([0.0001], length(y)))
end

s = sample(graph_mask(run_model(repeat([1], get(edge_index."shape", 1)), edge_index, model, node_idx)), 
        PG(5), 25000);

using DataFrames
@show DataFrame(describe(s)[1])

ei = edge_index."detach"()."cpu"()."numpy"()

base = run_model(repeat([1], get(edge_index."shape", 1)), edge_index, model, node_idx)
@show base/sum(base)

ŷ = run_model([x > 0.1 ? 1 : 0 for x ∈ DataFrame(describe(s)[1])[!, 2]], edge_index, model, node_idx)
@show ŷ/sum(ŷ)

using GraphRecipes
using Plots
using LightGraphs
theme(:juno)
gr(fmt=:svg)

uni = unique(ei);
conv = Dict(uni[i] => i for i ∈ 1:length(uni));
conv_rev = Dict(i => uni[i] for i ∈ 1:length(uni));
vals = DataFrame(describe(s)[1])[!, 2];
weigh = Dict((ei[1, i], ei[2, i]) => vals[i] for i ∈ 1:length(vals))

@show "Done 1"

ei_conv = map(x -> conv[x], ei)
inp = ei_conv
classes = y."detach"()."cpu"()."numpy"()[unique(map(x -> conv_rev[x], inp))]

@show "Done 2"

g = DiGraph(Edge.(zip(inp[1, :], inp[2, :])))

plot(graphplot(g, names=uni, arrow=true, nodecolor=classes, edgewidth=(s, d, w)-> 
    2*weigh[(conv_rev[s],conv_rev[d])], fontsize=12))
savefig("/gpfs_home/spate116/singhlab/GCN_Integration/notebooks/VI/out.png")
