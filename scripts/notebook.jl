### A Pluto.jl notebook ###
# v0.11.2

using Markdown
using InteractiveUtils

# ╔═╡ 6630175e-d5aa-11ea-3171-a1dd3b56dfda
using DrWatson

# ╔═╡ aa1da76a-d5aa-11ea-366d-efb23acf302a
begin
	@quickactivate "GCN_HM_GRN-Integration"
	using Plots, DataFrames
end

# ╔═╡ 1a72945c-d5ac-11ea-0c5d-53dd35d75bc3
begin
	theme(:default)
	gr(fmt = :png, dpi = 200)
end

# ╔═╡ b21e917c-d5aa-11ea-35e0-ff99bdaa8e2e
df = collect_results(datadir("inner_batch"));

# ╔═╡ dc048b40-d5aa-11ea-1ec2-75f9ac6c2592
begin
	plts = []
	for i ∈ 1:size(df, 1)
		plt = plot([df[i, :losses_train], df[i, :losses_test], df[i, :auc_list]], label=["train" "test" "auc"], legend = :outerright)
		title!(plt, "$(df[i, :layer]), $(df[i, :batch]), $(DrWatson.roundval(df[i, :auc]; digits = 4, scientific = 4))")
		pushfirst!(plts, plt)
	end
	p = plot(plts..., layout = (Int32(length(plts) / 2),2))
end

# ╔═╡ Cell order:
# ╠═6630175e-d5aa-11ea-3171-a1dd3b56dfda
# ╠═aa1da76a-d5aa-11ea-366d-efb23acf302a
# ╠═1a72945c-d5ac-11ea-0c5d-53dd35d75bc3
# ╠═b21e917c-d5aa-11ea-35e0-ff99bdaa8e2e
# ╠═dc048b40-d5aa-11ea-1ec2-75f9ac6c2592
