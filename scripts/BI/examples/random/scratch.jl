using Plots, GraphRecipes, GraphIO, ParserCombinator, LightGraphs
using GraphIO: GML

G = SimpleDiGraph(20)
weights = Dict()
lines = readlines("scripts/BI/random/explain/node1.el")
for line ∈ lines
    elts = split(line)
    s = parse(Int, elts[1]) + 1
    d = parse(Int, elts[2]) + 1
    add_edge!(G, s, d)
    weights[(s, d)] = parse(Float64, elts[3])
end

plt = plot(graphplot(G, names=0:19, arrow=true, nodecolor=fill(0, 20), edgewidth=(s, d, w)-> 
        weights[(s,d)], fontsize=8), size=(1000, 1000), dpi=300)
savefig("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/random/explain/node1.png")