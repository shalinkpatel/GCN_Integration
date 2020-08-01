using DrWatson
@quickactivate "GCN_HM_GRN-Integration"

using DataFrames, Latexify
df = collect_results(datadir("port_batched"));
rename!(df, ["cl", "train_loss", "test_loss", "layer", "auc_list",
                "lr", "best_auc", "loc"]);

using Plots
theme(:juno)
gr()

plts = []
for i ∈ 1:size(df, 1)
	plt = plot([df.auc_list[i], df.train_loss[i], df.test_loss[i]],
	        labels = ["auc" "train" "test"], legend = :right,
	        xlabel = "Epochs", title = "Training for Cell Line $(df.cl[i]) with lr = $(df.lr[i]) and $(df.layer[i]) layers (Best AUC: $(DrWatson.roundval(df.best_auc[i]; digits = 4, scientific = 4)))", size = (800, 500), dpi = 100)
	cl, layer, lr = df.cl[i], df.layer[i], df.lr[i]
	d = @dict cl layer lr
	push!(plts, plt)
end
p = plot(plts..., layout=(3, 1), size = (1000, 1500), dpi = 200)
savefig(p, plotsdir("port_batched", "summary.png"))