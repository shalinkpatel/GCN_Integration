using GraphIO, LightGraphs, JSON, Plots, GraphRecipes, ParserCombinator
using GraphIO: GML
gr(fmt=:svg)
theme(:juno)

function generate_syn(N)
    g = complete_graph(N)
    labels = Dict(i => 0 for i ∈ 1:N)
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

g, labels = generate_syn(20)
write_to_file(g, labels, 0)
