using DrWatson
@quickactivate "GCN_HM_GRN-Integration"
using Turing, PyCall, Zygote
using DataFrames
using GraphRecipes
using Plots
using LightGraphs
using Statistics
theme(:default)
gr(fmt=:svg)

Turing.setadbackend(:zygote)

using Zygote
Zygote.@adjoint function pycall(f, x...; kw...)
    x = map(py, x)
    y = pycall(f, x...; kw...)
    y.detach().numpy(), function (ȳ)
        y.backward(gradient = py(ȳ))
        (nothing, map(x->x.grad.numpy(), x)...)
    end
end

abstract type BayesianInferenceModel end

function load_pyg(path::String, file::String)
    pushfirst!(PyVector(pyimport("sys")."path"), path);
    train_model = pyimport(file).train_model
    run_model = pyimport(file).run_model
    extract_subgraph = pyimport(file).extract_subgraph
    pyimport("sys")."stdout" = PyTextIO(stdout)
    pyimport("sys")."stderr" = PyTextIO(stderr)
    return train_model, run_model, extract_subgraph
end

struct PyGInferenceModel <: BayesianInferenceModel
    C::Int32
    path::String
    file::String
    node::Int32
    k::Int32
    ei::Array{Int32, 2}
    base_pred::Array{Float32, 1}
    model
    x
    y
end

function PyGInferenceModel(path::String, file::String, node::Integer, k::Integer, p::Float64 = 0.1, stdev::Float64 = 0.001)
    node = convert(Int32, node)
    k = convert(Int32, k)
    train_model, run_model, extract_subgraph = load_pyg(path, file)
    model, x, y, edge_index = train_model();
    edge_index, node_idx = extract_subgraph(node, k, edge_index)
    edge_mask = repeat([1], get(edge_index."shape", 1));
    @model graph_mask(y) = begin
        N = get(edge_index."shape", 1)
        edge_mask = Vector{Int8}(undef, N)
        edge_mask ~ filldist(Bernoulli(p), N)
        pred = run_model(edge_mask, edge_index, model, node_idx)
        y = y/sum(y)
        y ~ MvNormal(pred/sum(pred), repeat([stdev], length(y)))
    end
    base = run_model(repeat([1], get(edge_index."shape", 1)), edge_index, model, node_idx)
    return PyGInferenceModel(get(edge_index."shape", 1), path, file, node, k, 
            edge_index."detach"()."cpu"()."numpy"(), base, graph_mask, x."detach"()."cpu"()."numpy"(), y."detach"()."cpu"()."numpy"())
end

function sample(m::T where T <: BayesianInferenceModel, iters::Integer)
    return Turing.sample(m.model(m.base_pred), PG(5), iters)
end

function final_summary(s::Chains)
    return DataFrame(describe(s)[1])
end

function plot_result(model::T where T <: BayesianInferenceModel, result::Chains, k::Number, dlpy::Bool = true, filter::Bool = true)
    ei = model.ei;
    vals = DataFrame(describe(result)[1])[!, 2];
    if filter
        sub_idx = (mean(vals) + k * var(vals)) .< vals;
        vals = vals[sub_idx]
        ei = ei[:, sub_idx]
    end
    
    weigh = Dict((ei[1, i], ei[2, i]) => vals[i] for i ∈ 1:length(vals));
    uni = unique(ei);
    conv = Dict(uni[i] => i for i ∈ 1:length(uni));
    conv_rev = Dict(i => uni[i] for i ∈ 1:length(uni));

    ei_conv = map(x -> conv[x], ei)
    inp = ei_conv
    classes = model.y[unique(map(x -> conv_rev[x], inp)) .+ 1]

    g = DiGraph(Edge.(zip(inp[1, :], inp[2, :])))

    plt = plot(graphplot(g, names=uni, arrow=true, nodecolor=classes, edgewidth=(s, d, w)-> 
        2*weigh[(conv_rev[s],conv_rev[d])], fontsize=8), size=(1000, 1000), dpi=300)
    if dlpy
        display(plt)
    end
    return plt
end
