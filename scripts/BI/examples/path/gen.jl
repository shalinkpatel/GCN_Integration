using LightGraphs, Plots, StatsPlots, Distributions, GraphRecipes, GraphIO, ParserCombinator, JSON
using GraphIO: GML
gr()
theme(:juno)

N = 50
k = 1

function gen_graph(N, k)
    G = barabasi_albert(N, k)

    path = a_star(G, 1, N)

    uni = Set{Int}()
    for e ∈ path
        push!(uni, e.src)
        push!(uni, e.dst)
    end

    labels = Dict(i => i in uni ? 1 : 0 for i in 1:N)
    return G, labels
end

function plot_syn_graph(g, labels)
    graphplot(g, nodecolor=map((x) -> labels[x] == 0 ? :green : :blue, 1:length(labels)))
end

function write_to_file(g, labels, n)
    open("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/path/data/syn_graph_labels_$(n).json", "w") do io
        write(io, JSON.json(labels))
    end;

    open("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/path/data/syn_graph_$(n).gml", "w") do io
        GML.savegml(io, g)
    end;
end

G, labels = gen_graph(N, k)
write_to_file(G, labels, 0)

for i in 1:length(labels)
    if labels[i] == 1
        print(i)
        print('\n')
    end
end