using Serialization, NPZ
include("model.jl")

node = 529
hops = 3
model = PyGInferenceModel("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/examples/syn/", 
                                "syn4", node, hops, 0.1f0, 0.0001f0)
@time s = sample(model, 25000)
@show final_summary(s)
serialize("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/hic_gcn/explain/samples$(node).jls", [s, model.ei, model.y])
plt = plot_result(model, s, 0, 2, true, false)
savefig("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/hic_gcn/explain/node$(node)BI.png")

edge_labels = npzread("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/examples/syn/data/syn4_edge_labels.npy")