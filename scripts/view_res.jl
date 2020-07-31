using DrWatson
@quickactivate "GCN_HM_GRN-Integration"

using DataFrames, Latexify
df = collect_results(datadir("port_no_sep"))
df = df[!, 1:4]
print(df)
copy_to_clipboard(true)
print(latexify(df; env=:table, latex=false))
