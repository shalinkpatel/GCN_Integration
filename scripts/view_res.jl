using DrWatson
@quickactivate "GCN_HM_GRN-Integration"

exp = "inner_batch"

using DataFrames, Latexify
df = collect_results(datadir(exp));

using Plots
theme(:juno)
gr()

plts = []
for i ∈ 1:size(df, 1)
	plt = plot([df.auc_list[i], df.losses_train[i], df.losses_test[i]],
	        labels = ["auc" "train" "test"], legend = :right,
	        xlabel = "Epochs", title = "Training for Cell Line $(df.cl[i]) with batch = $(df.batch[i]) and $(df.layer[i]) layers (Best AUC: $(DrWatson.roundval(df.auc[i]; digits = 4, scientific = 4)))", size = (800, 500), dpi = 100)
	push!(plts, plt)
end
p = plot(plts..., layout=(4, 3), size = (3200, 2000), dpi = 200)
savefig(p, plotsdir(exp, "summary.png"))
