using GraphIO, LightGraphs, JSON, Plots, GraphRecipes, ParserCombinator
using GraphIO: GML
gr(fmt=:svg)
theme(:juno)

function generate_syn(n₁, n₂, N, k)
    g = barbell_graph(n₁, n₂)
    barabasi_albert!(g, n₁ + n₂ + N, k)
    barabasi_albert!(g, n₁ + n₂ + 2*N, k)
    labels = Dict(i => i <= n₁ + n₂ ? 0 : 1 for i ∈ 1:(n₁ + n₂ + 2*N))
    return g, labels
end

function plot_syn_graph(g, labels)
    graphplot(g, nodecolor=map((x) -> labels[x] == 0 ? :green : :blue, 1:length(labels)))
end

function write_to_file(g, labels, n)
    open("Desktop/syn_graph_labels_$(n).json", "w") do io
        write(io, JSON.json(labels))
    end;

    open("Desktop/syn_graph_$(n).gml", "w") do io
        GML.savegml(io, g)
    end;
end

g, labels = generate_syn(8, 8, 4, 3)
write_to_file(g, labels, 3)
