using CSV, DataFrames, AlgebraOfGraphics, UMAP, Lazy, CairoMakie

df = CSV.File("100gene-9groups-20sparsity.csv") |> DataFrame
counts = df[:,2:end] |> Matrix
counts = Float64.(counts)
ys = @>> 1:2700 collect map(x -> (x-1) ÷ 300)

embedding = umap(counts', 2)

df = (x = embedding[1, :], y = embedding[2, :], c = map(x -> "$(x)", ys))
fig = draw(data(df) * mapping(:x, :y, color = :c, marker = :c))
save("mapping20.pdf", fig)
