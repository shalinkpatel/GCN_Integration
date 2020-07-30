using DrWatson
@quickactivate "GCN_HM_GRN-Integration"

using DataFrames
df = collect_results(datadir("port_no_sep"))
print(df[!, 1:4])
