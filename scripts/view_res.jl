using DrWatson
@quickactivate "GCN_HM_GRN-Integration"

using DataFrames, Latexify
df = collect_results(datadir("large_batch"));

using Plots
theme(:juno)
gr()

plts = []
for i ∈ 1:size(df, 1)
	plt = plot([df.auc_list[i], df.losses_train[i], df.losses_test[i]],
	        labels = ["auc" "train" "test"], legend = :right,
	        xlabel = "Epochs", title = "Training for Cell Line $(df.cl[i]) with batches = $(df.batch[i]) and $(df.layer[i]) layers (Best AUC: $(DrWatson.roundval(df.auc[i]; digits = 4, scientific = 4)))", size = (800, 500), dpi = 100)
	cl, layer, batches = df.cl[i], df.layer[i], df.batch[i]
	d = @dict cl layer batches
	push!(plts, plt)
end
p = plot(plts..., layout=(2, 2), size = (1600, 1000), dpi = 200)
savefig(p, plotsdir("large_batch", "summary.png"))