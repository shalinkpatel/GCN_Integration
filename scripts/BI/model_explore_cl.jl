using Serialization
include("model.jl")

node = 136736
hops = 0
model = PyGInferenceModel("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/hic_gcn/", 
                                "hic_gcn", node, hops, 0.05f0, 0.00025f0)
s = sample(model, 2000)
@show final_summary(s)
serialize("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/hic_gcn/explain/samples$(node).jls", [s, model.ei, model.y])
plt = plot_result(model, s, 1, true, true)
savefig("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/hic_gcn/explain/node$(node)BI.png")
