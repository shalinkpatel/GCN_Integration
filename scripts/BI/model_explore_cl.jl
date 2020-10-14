include("model.jl")

model = PyGInferenceModel("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/cora", 
                                "cora", 549, 2)
s = sample(model, 1000)
@show final_summary(s)
plt = plot_result(model, s, 0, false)
savefig("GCN_Integration/scripts/BI/cora/explain/node549BI.png")
display(plt)