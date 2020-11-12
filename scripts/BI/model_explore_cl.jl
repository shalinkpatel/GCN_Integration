include("model.jl")

model = PyGInferenceModel("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/path/", 
                                "path", 3, 2)
s = sample(model, 15000)
@show final_summary(s)
plt = plot_result(model, s, -1, true, false)
savefig("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/path/explain/node3BI.png")