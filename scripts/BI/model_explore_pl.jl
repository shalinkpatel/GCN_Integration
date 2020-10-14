### A Pluto.jl notebook ###
# v0.11.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ abc33bd6-ed31-11ea-33a4-313c8dd55164
using PlutoUI

# ╔═╡ ca921050-ed31-11ea-38f9-359d20b1b2ed
include("model.jl");

# ╔═╡ 9f27c4ee-ed36-11ea-0230-3f14526f9423
md"""
Node: $(@bind node Slider(range(1, 700, length = 1)))

k: $(@bind k Slider(1:3))

Samples: $(@bind samples Slider(range(100, 20000, length = 20)))

p₀: $(@bind p₀ Slider(range(0, 0.2, length = 21)))
"""

# ╔═╡ 1bac6944-ed3d-11ea-1a8f-0f0fb703e5ed
@show (node ,k, samples, p₀)

# ╔═╡ 438a5144-ed36-11ea-29df-7fa4eefea733
model = PyGInferenceModel("/gpfs_home/spate116/singhlab/GCN_Integration/scripts/BI/syn", "syn4", Int(node), k, p₀, 0.00000000000001);

# ╔═╡ f0eda5bc-ed37-11ea-3cc1-51dfea0858ad
@time s = sample(model, Int(samples));

# ╔═╡ 00e35172-ed38-11ea-0e7e-e5c07248c292
final_summary(s)

# ╔═╡ 0689b076-ed38-11ea-2f93-d70610e9b001
plt = plot_result(model, s)

# ╔═╡ 8133fb54-ed47-11ea-26c1-8b15400a881b
begin
	using Plots
	plot(plt, size=(1000, 1000), dpi=300)
	savefig("out.png")
end

# ╔═╡ Cell order:
# ╠═abc33bd6-ed31-11ea-33a4-313c8dd55164
# ╠═ca921050-ed31-11ea-38f9-359d20b1b2ed
# ╠═9f27c4ee-ed36-11ea-0230-3f14526f9423
# ╟─1bac6944-ed3d-11ea-1a8f-0f0fb703e5ed
# ╠═438a5144-ed36-11ea-29df-7fa4eefea733
# ╠═f0eda5bc-ed37-11ea-3cc1-51dfea0858ad
# ╠═00e35172-ed38-11ea-0e7e-e5c07248c292
# ╠═0689b076-ed38-11ea-2f93-d70610e9b001
# ╠═8133fb54-ed47-11ea-26c1-8b15400a881b
